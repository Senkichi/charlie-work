"""PR-diff fallback and unavailable-gate tests (issue #1807).

``PullRequests.pr_diff`` used to collapse every diff-fetch failure (GitHub's
300-file ``too_large`` cap, network blips, ...) to ``""``, indistinguishable
from a genuinely-empty diff, so ``review()`` wrote an empty ``diff.patch``
and the #1251 empty-diff gate skipped review dispatch forever. PR #2160 (a
461-file mechanical ledger split) hit exactly this, with >100
``review_dispatch_skipped_empty_diff`` warnings and no review ever run.

These tests cover the behavior seams:

- the ``PullRequests.pr_diff`` fallback (via ``make_github`` over a real
  local git checkout whose ``origin`` is a local bare repo serving
  ``refs/pull/N/head``, the same recipe ``git fetch origin pull/N/head``
  follows against GitHub.com);
- ``review()``'s refusal to build a packet when no diff could be fetched
  (FakeGitHub configured with ``diffs[N] = None``);
- the rework lane's routing of that refusal:
  ``_route_rework_candidate_to_review`` must not flip a
  ``rework_requested`` issue to ``reviewing`` over a packet that was never
  built (the refusal is ``ok=False``, matching the other no-packet retry
  exits).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _fake_transport import FakeAdapter, failure, make_github, ok
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig, RuntimeConfig
from charlie_work.github_capabilities.pr_diff_fallback import (
    base_ref_from_rest_pr,
)
from charlie_work.github_transport import FailureKind
from charlie_work.instrumentation import read_event_log
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_TOO_LARGE_BODY = {
    "message": (
        "Sorry, the diff exceeded the maximum number of files (300). "
        "Consider using 'List pull requests files' API or locally cloning "
        "the repository instead."
    ),
    "errors": [{"resource": "PullRequest", "field": "diff", "code": "too_large"}],
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.email=worker@example.test",
        "-c",
        "user.name=Worker",
        "commit",
        "-m",
        message,
    )


def _repo_with_pr_head(tmp_path: Path) -> Path:
    """A bare local ``origin`` (stand-in for GitHub.com) + a checkout that has
    pushed one base commit to ``main`` and one PR commit to
    ``refs/pull/7/head`` -- the ref ``git fetch origin pull/7/head`` reads."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "remote", "add", "origin", str(origin))
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _commit(repo, "base")
    _git(repo, "push", "-u", "origin", "main")
    (repo / "b.txt").write_text("new line\n", encoding="utf-8")
    _commit(repo, "pr head")
    _git(repo, "push", "origin", "HEAD:refs/pull/7/head")
    return repo


def _unavailable_event(state_file: Path) -> list[dict]:
    """Events of the #1807 kind from the state file's event log."""
    return [
        e for e in read_event_log(state_file) if e.get("kind") == "review_packet_diff_unavailable"
    ]


def test_base_ref_from_rest_pr_tolerates_non_pr_payloads() -> None:
    assert base_ref_from_rest_pr({"base": {"ref": "main"}}) == "main"
    assert base_ref_from_rest_pr({"base": {}}) is None
    assert base_ref_from_rest_pr({"base": {"ref": "  "}}) is None
    assert base_ref_from_rest_pr(None) is None
    assert base_ref_from_rest_pr([1, 2]) is None
    # Issue #1807 rework: the extracted ref is interpolated into
    # ``git diff --merge-base origin/<ref>`` argv, so a malformed or hostile
    # name must fail ``safe_ref.require_valid_ref_name`` and read as "cannot
    # determine the base branch" -- never as a rev-range or a parsed flag.
    assert base_ref_from_rest_pr({"base": {"ref": "main..HEAD"}}) is None
    assert base_ref_from_rest_pr({"base": {"ref": "--flag"}}) is None


def test_pr_diff_too_large_served_from_local_git_fallback(tmp_path: Path) -> None:
    """GitHub's 300-file ``too_large`` diff cap (PR #2160: 461 files) must not
    produce an empty diff: the local-git fallback fetches ``pull/7/head`` and
    renders the same merge-base diff, and the success is NOT an event."""
    repo = _repo_with_pr_head(tmp_path)
    http = FakeAdapter(
        "http",
        [
            ok(_TOO_LARGE_BODY, status=403),
            ok({"base": {"ref": "main"}, "number": 7}),
        ],
    )
    github, http_adapter, _ = make_github(repo, http=http)

    result = github.pr_diff(7)

    assert result is not None
    assert "diff --git" in result
    assert "new line" in result
    # One GET with the diff media type, then the baseRefName read.
    assert len(http_adapter.api_requests) == 2
    # A served diff is not an unavailable event.
    rt = RuntimeConfig()
    state_file = runtime_paths(repo, rt.state_dir).state_file
    assert _unavailable_event(state_file) == []


def test_pr_diff_unavailable_returns_none_and_emits_warning(tmp_path: Path) -> None:
    """A diff GitHub refuses to serve AND whose local fallback fails returns
    ``None`` -- never ``""`` -- and emits the ``review_packet_diff_unavailable``
    warning naming the transport error (the text the #1807 report showed)."""
    repo = _repo_with_pr_head(tmp_path)
    # The baseRefName read fails too (adapter defect: not a retryable
    # outcome, so the guarded transport does not loop on it, but it DOES fall
    # back to the gh adapter -- script that one the same way so the call
    # concludes as a single transport failure).
    http = FakeAdapter(
        "http",
        [
            ok(_TOO_LARGE_BODY, status=403),
            failure(FailureKind.ADAPTER_DEFECT, "malformed body"),
        ],
    )
    gh_alias = FakeAdapter(
        "gh",
        [failure(FailureKind.ADAPTER_DEFECT, "malformed body")],
        token="tok-1",
    )
    github, http_adapter, _ = make_github(repo, http=http, gh=gh_alias)

    result = github.pr_diff(7)

    assert result is None
    assert len([r for r in http_adapter.api_requests]) == 2
    rt = RuntimeConfig()
    state_file = runtime_paths(repo, rt.state_dir).state_file
    events = _unavailable_event(state_file)
    assert len(events) == 1
    error = events[0]["payload"]["error"]
    assert "exceeded the maximum number of files (300)" in error
    assert events[0]["payload"]["pr_number"] == 7


def test_pr_diff_empty_success_keeps_empty_string(tmp_path: Path) -> None:
    """A *successfully fetched* empty diff stays ``""``: the #1251 gate's
    skip-forever behavior for a zero-delta PR is preserved, and no
    unavailable event fires."""
    github, _, _ = make_github(tmp_path, http=FakeAdapter("http", [ok("")]))
    result = github.pr_diff(9)
    assert result == ""
    state_file = runtime_paths(tmp_path, RuntimeConfig().state_dir).state_file
    assert _unavailable_event(state_file) == []


def test_review_refuses_to_build_packet_when_diff_unavailable(tmp_path: Path) -> None:
    """A PR whose diff fetch failed everywhere must not produce a packet with
    an empty diff.patch (which the #1251 gate reads as a zero-delta PR and
    never dispatches): no packet files, no review-started transition, and a
    result the next pass retries. The refusal is ``ok=False`` -- matching
    ``head_moved_during_build`` and the other no-packet retry exits in
    ``review()`` -- so no rework routing can read it as a fresh packet."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        '{"version": 1, "issues": {}, "prs": {}, "events": []}', encoding="utf-8"
    )
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = None
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert "diff unavailable" in result.message
    pr_dir = paths.prs / "pr-456"
    assert pr_dir.exists() is False
    assert fake_gh.labels_added == []
    # The refusal carries the discriminator the reworks/escalation paths read.
    assert result.data.get("reason") == "diff_unavailable"


def test_route_rework_candidate_does_not_flip_when_diff_unavailable(
    tmp_path: Path,
) -> None:
    """Issue #1807 rework: a ``rework_requested`` issue whose PR diff is
    unavailable must NOT be flipped to ``reviewing`` when
    ``_route_rework_candidate_to_review`` routes it to review() (issue
    #339 finding 1's desync, one exit over): the flip condition requires a
    fresh packet, and ``review()`` built none -- its diff_unavailable exit
    is a no-packet retry (``ok=False``), so the issue stays
    ``rework_requested`` and the next dispatch_rework pass retries."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.diffs[456] = None
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    routed, review_result = app._route_rework_candidate_to_review(123, 456, "sha-abc123")

    # review() refused at the #1807 gate: not a fresh packet, not a routing.
    assert review_result.ok is False
    assert review_result.data.get("reason") == "diff_unavailable"
    assert routed is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    pushed = [e for e in state.get("events", []) if e.get("kind") == "rework_already_pushed"]
    assert len(pushed) == 1
    assert pushed[0]["payload"]["routed"] is False
    # No packet was built behind the routing's back either.
    assert (paths.prs / "pr-456").exists() is False
    assert fake_gh.labels_added == []
