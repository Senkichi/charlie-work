"""GitHub mutating calls: label add/remove, branch delete, dry-run mutating-command guard, token-mint retry.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from charlie_work import github as github_module


def test_github_delete_branch_failure_returns_false(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="Reference does not exist",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    assert github_module.GitHub(tmp_path).delete_branch("agent/issue-1-x") is False


def test_github_add_issue_label_failure_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    """C5 boundary test: add_issue_label with allow_failure=True returns error value, does not raise."""

    def fake_run(cmd, *args, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="simulated failure")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Should not raise despite subprocess failure (allow_failure=True in add_issue_label)
    gh.add_issue_label(123, "agent:in-progress")


def test_github_remove_issue_label_failure_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    """C5 boundary test: remove_issue_label with allow_failure=True returns error value, does not raise."""

    def fake_run(cmd, *args, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="simulated failure")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Should not raise despite subprocess failure (allow_failure=True in remove_issue_label)
    gh.remove_issue_label(123, "agent:in-progress")


def test_github_add_issue_label_returns_false_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns False on subprocess failure (returncode=1)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is False, "add_issue_label must return False on failure"


def test_github_add_issue_label_returns_true_on_success(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns True on subprocess success (returncode=0)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is True, "add_issue_label must return True on success"


def test_github_remove_issue_label_returns_false_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns False on subprocess failure (returncode=1)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is False, "remove_issue_label must return False on failure"


def test_github_remove_issue_label_returns_true_on_success(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns True on subprocess success (returncode=0)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is True, "remove_issue_label must return True on success"


def test_github_dry_run_skips_mutating_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path, dry_run=True)

    out = gh.run(["pr", "merge", "1", "--squash"])

    assert out.startswith("DRY-RUN:")
    assert calls == []  # subprocess.run never invoked for a mutating command


def test_github_dry_run_allows_readonly_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path, dry_run=True)

    gh.run(["issue", "list", "--label", "x"], json_output=True)

    assert len(calls) == 1  # read-only command still executes under dry-run


def test_is_mutating_classifies_readonly_and_mutating() -> None:
    from charlie_work.github import _is_mutating

    for readonly in (
        ["issue", "list"],
        ["pr", "view", "1"],
        ["pr", "checks", "1"],
        ["label", "list"],
    ):
        assert _is_mutating(readonly) is False
    for mutating in (["pr", "merge", "1"], ["issue", "edit", "1"], ["label", "create", "x"]):
        assert _is_mutating(mutating) is True


def test_is_mutating_blocks_the_argv_delete_branch_actually_builds(
    monkeypatch, tmp_path: Path
) -> None:
    """#914/#917: `-X DELETE` classified as read-only, so `--dry-run` really deleted
    PR head branches.

    The argv is captured from `delete_branch` itself rather than written out by hand,
    so the gate cannot drift away from its most destructive caller: if someone changes
    how that call is spelled, this test follows it.
    """
    from charlie_work.github import _is_mutating

    captured: list[list[str]] = []
    # GitHub is a frozen dataclass -- patch the class, not the instance.
    monkeypatch.setattr(
        github_module.GitHub,
        "run",
        lambda self, args, **kwargs: captured.append(args) or "",
    )
    gh = github_module.GitHub(repo_root=tmp_path)

    assert gh.delete_branch("feature/x") is True
    assert len(captured) == 1
    assert _is_mutating(captured[0]) is True


def test_is_mutating_api_method_spellings_and_preserved_reads() -> None:
    """Every spelling `gh` accepts for a method must classify from that method, and
    an unparseable one must fail CLOSED.

    The read-only half is the half that protects `--dry-run` from being tightened into
    uselessness: these are real live call sites, and without them a future "just deny
    all `gh api`" change passes every other test in the suite.
    """
    from charlie_work.github import _is_mutating

    for mutating in (
        ["api", "-X", "DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X=DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-XDELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X", "POST", "repos/o/r/actions/runners/remove-token"],
        ["api", "--method", "PATCH", "repos/o/r/issues/1"],
        ["api", "--method=PUT", "repos/o/r/branches/main/protection"],
        ["api", "-X"],  # named but valueless -> fail closed, not open
        ["api", "--method"],
        ["api", "repos/o/r/issues", "-f", "title=x"],  # params switch gh to POST
        ["api", "repos/o/r/issues", "--field=labels[]=bug"],
        # pflag takes an attached shorthand value here too, exactly as for -X (#919).
        ["api", "repos/o/r/issues", "-ftitle=x"],
        ["api", "repos/o/r/issues", "-Flabels[]=bug"],
        ["api", "repos/o/r/issues", "-F"],
        ["api", "repos/o/r/issues", "--raw-field", "title=x"],
        ["api", "repos/o/r/issues", "--input", "body.json"],
        ["api", "repos/o/r/issues", "--input=-"],
    ):
        assert _is_mutating(mutating) is True, mutating

    for readonly in (
        ["api", "rate_limit"],
        ["api", "repos/o/r/commits/abc/check-runs"],
        ["api", "repos/o/r/compare/main...topic"],
        ["api", "repos/o/r/branches/main/protection"],
        ["api", "-X", "GET", "repos/o/r/issues"],
        ["api", "--method=HEAD", "repos/o/r"],
        # A header is not a method -- github.py:1033 fetches a diff this way.
        ["api", "repos/o/r/pulls/1", "-H", "Accept: application/vnd.github.v3.diff"],
        # Other shorthands must not be swept up by the -f/-F prefix match (#919).
        ["api", "repos/o/r/issues", "-q", ".[].number"],
        ["api", "repos/o/r/issues", "-t", "{{.number}}"],
        ["api", "repos/o/r/issues", "--paginate"],
    ):
        assert _is_mutating(readonly) is False, readonly


def test_token_minting_posts_do_not_retry_post_send_failures() -> None:
    """Credential-minting POSTs must retry only on provable pre-send failures.

    `_is_mutating` has a second consumer besides the --dry-run gate: `run()` feeds it
    to `_should_retry`, which grants reads an unconditional retry on any transient
    error and restricts mutations to pre-connection errors, so a request that may
    already have been applied is never re-sent.

    Before #918 these two argvs classified as *reads* (the `-X` spelling fell through
    the old enumeration), which put credential minting on the unconditional-retry
    path: a post-send timeout on a request GitHub had actually served would mint a
    second token. #918 fixed that as a side effect of the --dry-run work without
    naming it, so pin it here -- a future reclassification of `-X POST` would
    otherwise reopen the loop with every other test still green (#919).
    """
    from charlie_work.github import _is_mutating, _should_retry

    for argv in (
        ["api", "-X", "POST", "repos/{owner}/{repo}/actions/runners/remove-token"],
        [
            "api",
            "-X",
            "POST",
            "repos/{owner}/{repo}/actions/runners/registration-token",
        ],
    ):
        assert _is_mutating(argv) is True, argv
        # Ambiguous: the request may have been served before the read timed out.
        assert _should_retry(argv, "i/o timeout", _is_mutating(argv)) is False, argv
        # Provably pre-send -- no token can have been minted, so retrying is safe.
        assert _should_retry(argv, "dial tcp: connection refused", _is_mutating(argv)) is True, (
            argv
        )

    # Control: a read is still granted the unconditional retry, so the assertions
    # above are about the mutating classification and not about the error strings.
    read = ["api", "repos/{owner}/{repo}/actions/runners"]
    assert _should_retry(read, "i/o timeout", _is_mutating(read)) is True
