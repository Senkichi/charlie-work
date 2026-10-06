"""``self_deploy`` pull/sync tests plus its source-root anchor.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``. The deferral-gate cluster lives in
``tests/test_supervise_self_deploy_deferral.py`` (file-size ratchet split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _supervise_fixtures import (
    _make_fake_runner,
    no_fleet_live_sessions as no_fleet_live_sessions,
)
from charlie_work.git_retry import RetryOutcome
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    SelfDeployResult,
    _command_failure_message,
    _log_self_deploy_git_retry,
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
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
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
    assert len(calls) == 6
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin", "main"],
        ["git", "rev-parse", "origin/main"],
        ["git", "merge-base", "--is-ancestor", "def456", "HEAD"],
        ["git", "diff", "--name-only", "abc123..def456"],
        ["git", "merge", "--ff-only", "origin/main"],
    ]


def test_self_deploy_dependency_change_triggers_uv_sync(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A pull touching pyproject.toml/uv.lock runs uv sync --locked --inexact
    and reports success."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked --inexact ok
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
    assert calls[-1][0] == ["uv", "sync", "--locked", "--inexact"]


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
    the six the merge path itself consumed (the pre-merge ancestor gate being
    the sixth).
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # ancestor gate: target not an ancestor of HEAD
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
    assert len(calls) == 9


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
    assert all(c[0] != ["uv", "sync", "--locked", "--inexact"] for c in calls)


def test_self_deploy_uv_sync_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """If uv sync fails after a dependency-changing merge, self_deploy reports the error."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
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
            RunResult(1, "", ""),  # ancestor gate: target not an ancestor of HEAD
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
    assert len(calls) == 9


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
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
            RunResult(0, "", ""),  # merge --ff-only ok
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.pulled is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert len(calls) == 7
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
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
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
