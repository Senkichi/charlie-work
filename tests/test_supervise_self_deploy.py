"""``self_deploy`` pull/sync/deferral tests plus its source-root anchor.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _supervise_fixtures import (
    _make_fake_runner,
    no_fleet_live_sessions as no_fleet_live_sessions,
)
from charlie_work import layout
from charlie_work.git_retry import RetryOutcome
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    SelfDeployResult,
    _command_failure_message,
    _log_self_deploy_git_retry,
    _parse_marker_timestamp,
    _pending_sync_marker_path,
    _self_deploy_state_path,
    orchestrator_root,
    self_deploy,
)


def test_self_deploy_code_only_change_does_not_sync(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A pull that changes only source files triggers no uv sync."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "src/foo.py\nREADME.md\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only ok
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        head_changed=True,
        from_sha="abc123",
        to_sha="def456",
        message="code-only update: def456",
    )
    assert len(calls) == 5
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin", "main"],
        ["git", "rev-parse", "origin/main"],
        ["git", "diff", "--name-only", "abc123..def456"],
        ["git", "merge", "--ff-only", "origin/main"],
    ]


def test_self_deploy_dependency_change_triggers_uv_sync(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A pull touching pyproject.toml/uv.lock runs uv sync --locked and reports success."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked ok
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert "updated and synced" in result.message
    assert ["git", "merge", "--ff-only", "origin/main"] in [c[0] for c in calls]
    assert calls[-1][0] == ["uv", "sync", "--locked"]


def test_self_deploy_pull_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A network-shaped fetch refusal fails the pass without raising.

    The leaf name says ``pull`` deliberately: at the deploy level the pull
    step IS the fetch -- a fetch refusal is the failure that leaves
    ``result.pulled`` False, exactly what the pre-#2312 fused ``git pull``
    refusal produced. The name is also load-bearing for the collect-only
    gate (issue #1538), which is fail-closed on test renames.

    Fetch failure never reaches the lossless-blocker repair -- that repair is
    for merge refusals (the tree is in the way of incoming blobs), and with a
    failed fetch there is no new origin/main to be in the way of.
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(1, "", "fatal: could not read from remote repository."),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.pulled is False
    assert result.changed is False
    assert result.synced is False
    assert result.from_sha == "abc123"
    assert "could not read from remote" in (result.error or "")
    assert len(calls) == 2


def test_self_deploy_merge_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A diverged/dirty tree makes the merge fail; self_deploy returns but does not raise.

    On failure, ``_self_deploy_attempt`` calls ``_repair_lossless_pull_blockers``
    before giving up, which re-reads HEAD and ``origin/main`` and, in a genuinely
    diverged tree, bails out at the ``merge-base --is-ancestor`` check -- three
    extra canned responses (HEAD, origin/main, the ancestor check failing) beyond
    the five the merge path itself consumed.
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
            RunResult(1, "", "fatal: Not possible to fast-forward, aborting."),
            RunResult(0, "abc123\n", ""),  # repair: rev-parse HEAD
            RunResult(0, "def999\n", ""),  # repair: rev-parse origin/main (diverged)
            RunResult(1, "", ""),  # repair: merge-base --is-ancestor fails
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is False
    assert result.from_sha == "abc123"
    assert "fast-forward" in (result.error or "")
    assert len(calls) == 8


def test_self_deploy_already_up_to_date(tmp_path: Path, no_fleet_live_sessions: None) -> None:
    """When the fetch succeeds but origin/main does not move, no sync is attempted."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "From https://x\n", ""),  # fetch ok
            RunResult(0, "abc123\n", ""),  # origin/main (same as HEAD)
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=False,
        synced=False,
        head_changed=False,
        from_sha="abc123",
        to_sha="abc123",
        message="already up to date",
    )
    assert len(calls) == 3
    assert all(c[0] != ["uv", "sync", "--locked"] for c in calls)


def test_self_deploy_uv_sync_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """If uv sync fails after a dependency-changing merge, self_deploy reports the error."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "uv.lock\n", ""),  # dep file changed
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(1, "", "failed to install"),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is False
    assert result.to_sha == "def456"
    assert "failed to install" in (result.error or "")


def test_self_deploy_defers_sync_when_fleet_runners_active(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A dependency-changing update defers BEFORE the merge while fleet live
    sessions are active (issue #2312).

    The deferral decision now sits between the fetch and the merge, so a
    deferred pass leaves HEAD parked at ``before_sha`` -- the checkout's
    source and its installed venv stay consistent, and a supervisor
    restarted mid-deferral cannot crash on missing deps. ``head_changed``
    is therefore False here (previously True: the old ordering merged first
    and parked a *broken* environment).
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (should not be reached)
        ]
    )

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 2, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        head_changed=False,
        from_sha="abc123",
        to_sha="def456",
        message="sync deferred: 2 runners active",
        deferred=True,
    )
    assert all(c[0] != ["uv", "sync", "--locked"] for c in calls)
    # Issue #2312: the merge must never have been issued -- HEAD stays parked.
    assert all(c[0][:2] != ["git", "merge"] for c in calls)
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin", "main"],
        ["git", "rev-parse", "origin/main"],
        ["git", "diff", "--name-only", "abc123..def456"],
    ]

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["from_sha"] == "abc123"
    assert marker["to_sha"] == "def456"
    # Issue #1855: the first deferral of an episode stamps ``written_at`` --
    # the timestamp the starvation bound measures -- and a fresh episode has
    # no ``starved_notified`` latch.
    assert _parse_marker_timestamp(marker["written_at"]) is not None
    assert "starved_notified" not in marker


def test_self_deploy_deferred_marker_replay_defers_again_without_merging(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A deferred pass holds HEAD parked; the next pass replays the marker.

    Issue #2312's companion invariant: once a deferral has parked HEAD below
    the marker's ``to_sha``, subsequent passes must keep holding the merge --
    not just the sync -- until the fleet drains. The marker from a pre-#2312
    deploy (HEAD already at ``to_sha``, covered by
    ``test_self_deploy_loud_warning_on_repeated_deferral``) still defers on
    the same ``live_count > 0`` gate; only the shape of what is held differs.
    """
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"from_sha": "abc123", "to_sha": "def456"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (1, []),
    )
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD still parked at from_sha
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "uv.lock\n", ""),  # diff
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.deferred is True
    assert result.head_changed is False
    assert all(c[0][:2] != ["git", "merge"] for c in calls)
    assert all(c[0] != ["uv", "sync", "--locked"] for c in calls)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["to_sha"] == "def456"


def test_self_deploy_honors_state_root_override(tmp_path: Path, monkeypatch: Any) -> None:
    """Issue #720: a configured state_root moves the pending-sync marker out of default."""
    custom_state_root = tmp_path / ".var" / "devin-orchestrator"

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 1, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
        ]
    )

    result = self_deploy(tmp_path, state_root=custom_state_root, run_command=runner)
    assert result.deferred is True
    assert _pending_sync_marker_path(custom_state_root).exists()
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_proceeds_when_zero_fleet_runners(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A dependency-changing pull merges and runs uv sync when no fleet live
    sessions are active."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked ok
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is True
    assert result.head_changed is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert "updated and synced" in result.message
    assert calls[-1][0] == ["uv", "sync", "--locked"]
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_retries_sync_after_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deferred dependency sync is retried on the next pass once runners are idle.

    Post-#2312 shape: pass N parks HEAD at ``from_sha`` (the merge is held,
    not just the sync), so pass N+1 replays the marker *and* lands the held
    merge before ``uv sync --locked`` -- the deferral covers the whole
    "advance checkout + sync env" transition as one unit.
    """
    live_counts = iter([2, 0])

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return next(live_counts), []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    # Pass N: dependency-changing update, two active runners -> defer before
    # the merge; HEAD stays at abc123 and the marker records the held range.
    first_runner, first_calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge/uv sync (not reached)
        ]
    )

    first = self_deploy(tmp_path, run_command=first_runner)
    assert first.synced is False
    assert first.head_changed is False
    assert first.message == "sync deferred: 2 runners active"

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()

    # Pass N+1: HEAD still parked at abc123, no new origin commits, runners
    # now idle -> merge the held range, then sync from marker and clear it.
    second_runner, second_calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD (parked)
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main (unchanged)
            RunResult(0, "uv.lock\n", ""),  # diff abc123..def456
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked ok
        ]
    )

    second = self_deploy(tmp_path, run_command=second_runner)
    assert second == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=True,
        head_changed=True,
        from_sha="abc123",
        to_sha="def456",
        message="updated and synced: def456",
    )
    assert ["git", "merge", "--ff-only", "origin/main"] in [c[0] for c in second_calls]
    assert second_calls[-1][0] == ["uv", "sync", "--locked"]
    assert not marker_path.exists()


def test_self_deploy_loud_warning_on_repeated_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a pending-sync marker survives repeated passes, a warning is printed.

    Also the outage regression guard: HEAD did not move on *this* attempt
    (before == after == "def456") even though a marker from an earlier
    deferral is still present and live workers are still active. Callers
    must see ``head_changed is False`` here -- gating a watchdog restart on
    ``from_sha != to_sha`` instead (the marker's original range, from
    "abc123" to "def456") previously caused the supervisor to exit and
    relaunch every single pass without ever reaching zero live workers to
    complete the deferred sync (the total-fleet-outage bug this test guards
    against).
    """
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (3, []),
    )

    # Create marker from a previous deferral.
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"from_sha": "abc123", "to_sha": "def456"}), encoding="utf-8"
    )

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # HEAD (already at the marker's to_sha
            # -- the pre-#2312 residue shape: the deploy merged but the sync
            # was deferred, so only the marker replays)
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main (unchanged)
            RunResult(0, "", ""),  # uv sync (not reached)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.synced is False
    assert result.head_changed is False
    assert "3 runners active" in result.message
    assert marker_path.exists()

    out = capsys.readouterr().out
    assert "WARNING: pending dependency sync still deferred" in out
    assert "3 runners active" in out
    assert "abc123..def456" in out


def test_self_deploy_pull_failure_surfaces_stderr_over_generic_error(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 3: a realistic failed RunResult -- as ``run_captured``
    actually produces on any non-zero exit, with ``.error`` always populated
    with the generic ``"command exited N"`` rather than left at the
    dataclass default ``None`` -- must still surface git's specific stderr
    (which names the colliding path) instead of the uninformative generic
    message.

    ``test_self_deploy_pull_failure_is_non_fatal`` above never caught the
    old ``result.error or result.stderr`` bug because it constructs
    ``RunResult(1, "", "fatal: ...")`` without ``.error``, leaving it at the
    ``None`` default -- under the old fallback chain that made ``.stderr``
    win "by accident" (None is falsy), masking the real production
    shadowing where ``.error`` is always truthy.

    Since #2312 the colliding-worktree refusal comes from
    ``git merge --ff-only`` (the fetch half already succeeded), and the
    trailing three canned responses account for
    ``_repair_lossless_pull_blockers`` short-circuiting at the
    diverged-tree check -- see the sibling test above for the same shape.
    The leaf name predates the split and is load-bearing for the
    collect-only gate (issue #1538): ``pull`` denotes the deploy's update
    step, whose merge leg is where this refusal now surfaces.
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
            RunResult(
                returncode=1,
                stdout="",
                stderr=(
                    "error: Your local changes to the following files would be "
                    "overwritten by merge:\n\tsrc/charlie_work/config.py\n"
                    "Please commit your changes or stash them before you merge."
                ),
                error="command exited 1",
            ),  # merge --ff-only refuses
            RunResult(0, "abc123\n", ""),  # repair: rev-parse HEAD
            RunResult(0, "def999\n", ""),  # repair: rev-parse origin/main (diverged)
            RunResult(1, "", ""),  # repair: merge-base --is-ancestor fails
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.error is not None
    assert "src/charlie_work/config.py" in result.error
    assert "command exited 1" not in result.error
    assert result.error.startswith("git merge --ff-only origin/main: ")
    assert len(calls) == 8


def test_self_deploy_pull_retries_transient_failure_then_succeeds(
    tmp_path: Path, no_fleet_live_sessions: None, monkeypatch: Any
) -> None:
    """A transient git-network blip on the fetch is retried in place (raw
    ``git`` calls previously had zero retry, unlike ``GitHub.run()``'s ``gh``
    calls) rather than failing the whole pass -- and exactly one
    ``git_network_retry`` event lands in events.db, not one per attempt.

    If the fetch call site were reverted to a bare ``run_command`` call (no
    ``run_git_with_retry`` wrapping), this test fails: the fake runner would
    hand the transient-failure ``RunResult`` straight back as the fetch's
    final result instead of retrying, and the queued "origin/main"/"diff"
    responses would never be consumed. The leaf name says ``pull`` because
    the retried leg is the pull's fetch -- and because the collect-only
    gate (issue #1538) is fail-closed on renames.

    ``git_retry``'s own ``time.sleep`` is monkeypatched out (issue #1777
    finding 8): ``self_deploy`` -> ``run_git_with_retry`` does not expose a
    ``sleep=`` seam of its own, so without this the real ~0.75-1.25s backoff
    would run for real on every pass of this test.
    """
    import charlie_work.git_retry as git_retry_module

    monkeypatch.setattr(git_retry_module.time, "sleep", lambda _seconds: None)

    state_path = _self_deploy_state_path(tmp_path)
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(
                returncode=128,
                stdout="",
                stderr=(
                    "fatal: unable to access 'https://github.com/x/y.git/': Failed to "
                    "connect to github.com port 443 after 2093 ms: Couldn't connect to "
                    "server"
                ),
            ),  # fetch attempt 1: transient (real curl/schannel shape, not the
            # hand-assembled `connectex` hybrid no tool actually emits --
            # issue #1777 finding 2)
            RunResult(0, "", ""),  # fetch attempt 2 (retry): ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
            RunResult(0, "", ""),  # merge --ff-only ok
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.pulled is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert len(calls) == 6
    assert calls[1][0] == ["git", "fetch", "origin", "main"]

    retries = query_events(state_path, kind="git_network_retry")
    assert len(retries) == 1
    # site is call-site-specific (issue #1777 finding 4), not a shared
    # "self_deploy" literal indistinguishable from the ci-fleet sibling pull.
    assert retries[0]["payload"]["site"] == "self_deploy_fetch"
    assert retries[0]["payload"]["cwd"] == str(tmp_path)
    assert retries[0]["payload"]["attempts"] == 2
    assert retries[0]["payload"]["ok"] is True


def test_log_self_deploy_git_retry_site_and_cwd_distinguish_call_sites(
    tmp_path: Path,
) -> None:
    """Issue #1777 finding 4: the orchestrator's own fetch and the ci-fleet
    sibling pull both log to this one state path and (before this fix)
    shared a hardcoded ``site="self_deploy"`` -- indistinguishable in
    events.db. ``site`` is now a required keyword-only parameter (not a
    default), and ``cwd`` carries the checkout the command actually ran in.

    Calling with the old two-positional-argument signature would now raise
    ``TypeError`` (missing keyword-only ``site``), which is itself a strong
    signal this test would fail against the pre-fix code -- confirmed by
    inspection of the pre-fix signature (``repo_root, command, outcome``
    only).
    """
    sibling = tmp_path / "ci-fleet"
    for command, site, cwd in (
        (["git", "fetch", "origin", "main"], "self_deploy_fetch", tmp_path),
        (["git", "pull", "--ff-only", "origin", "main"], "ci_fleet_sibling_pull", sibling),
    ):
        _log_self_deploy_git_retry(
            tmp_path,
            command,
            RetryOutcome(attempts=2, ok=True, error=None),
            site=site,
            cwd=cwd,
        )

    events = query_events(_self_deploy_state_path(tmp_path), kind="git_network_retry")
    assert [e["payload"]["site"] for e in events] == [
        "self_deploy_fetch",
        "ci_fleet_sibling_pull",
    ]
    assert [e["payload"]["cwd"] for e in events] == [str(tmp_path), str(sibling)]


def test_command_failure_message_falls_back_to_error_then_fallback() -> None:
    """``_command_failure_message`` prefers stderr, then .error, then fallback."""
    stderr_result = RunResult(1, "", "  stderr detail  ")
    assert _command_failure_message(["git", "pull"], stderr_result, "pull failed") == (
        "git pull: stderr detail"
    )

    error_only_result = RunResult(returncode=None, stdout="", stderr="", error="boom")
    assert _command_failure_message(["uv", "sync"], error_only_result, "sync failed") == (
        "uv sync: boom"
    )

    empty_result = RunResult(returncode=1, stdout="", stderr="")
    assert _command_failure_message(["git", "diff"], empty_result, "diff failed") == (
        "git diff: diff failed"
    )


def test_self_deploy_records_events_db_outcome_for_every_pass(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 4: every real self_deploy pass -- success, skip, and
    failure -- is durably recorded to events.db, queryable via
    ``query_events``. Before this fix, self_deploy had zero events.db
    instrumentation; the live fleet accumulated 121 consecutive real deploy
    failures with zero rows to show for it.
    """
    state_path = _self_deploy_state_path(tmp_path)

    # Pass 1: code-only success.
    runner1, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(0, "src/foo.py\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only
        ]
    )
    self_deploy(tmp_path, run_command=runner1)

    # Pass 2: already up to date -- ok, but nothing changed (a skip).
    runner2, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),
            RunResult(0, "", ""),  # fetch
            RunResult(0, "def456\n", ""),  # origin/main (same)
        ]
    )
    self_deploy(tmp_path, run_command=runner2)

    # Pass 3: fetch failure. A failed fetch never reaches the lossless
    # blocker repair (there is no new origin/main to be in the way of), so
    # this pass consumes exactly two canned responses.
    runner3, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),
            RunResult(
                returncode=1,
                stdout="",
                stderr="fatal: could not read from remote repository.",
                error="command exited 1",
            ),
        ]
    )
    self_deploy(tmp_path, run_command=runner3)

    succeeded = query_events(state_path, kind="self_deploy_succeeded")
    skipped = query_events(state_path, kind="self_deploy_skipped")
    failed = query_events(state_path, kind="self_deploy_failed")
    assert len(succeeded) == 1
    assert len(skipped) == 1
    assert len(failed) == 1
    assert failed[0]["level"] == "error"
    assert "could not read from remote" in failed[0]["payload"]["error"]

    # query_events(level="error") -- the general-purpose alerting query
    # already used elsewhere in the codebase -- surfaces the failure without
    # any self-deploy-specific query infrastructure.
    errors = query_events(state_path, level="error")
    assert any(e["kind"] == "self_deploy_failed" for e in errors)


def test_self_deploy_preview_does_not_record_events_db(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A ``dry_run`` preview touches nothing, including events.db (item 4)."""
    state_path = _self_deploy_state_path(tmp_path)
    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner, dry_run=True)
    assert result.previewed is True
    assert query_events(state_path, kind="self_deploy_succeeded") == []
    assert query_events(state_path, kind="self_deploy_skipped") == []
    assert query_events(state_path, kind="self_deploy_failed") == []


def test_orchestrator_root_contains_pyproject_toml() -> None:
    """orchestrator_root() resolves to the orchestrator source tree root."""
    root = orchestrator_root()
    assert (root / "pyproject.toml").is_file()
    # It should be the directory that holds the source tree, not a subpackage.
    assert (root / "src" / "charlie_work" / "supervise.py").is_file()
