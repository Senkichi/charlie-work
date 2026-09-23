"""Review-mode launch tests for ``devin_shell.launch_devin_session``.

Direct unit coverage of the ``review=True`` branch (issue #1536, follow-up
to the adversarial review of PR #1535 / issue #1513 role-harness symmetry):
``create_review_checkout`` kwargs forwarding, the session sidecar a review
launch leaves in the caller's ``reviews_dir`` (readable by
``worker.iter_workers`` as ``adapter_kind == "devin"``), and
``remove_review_checkout`` teardown on the spawn-failure path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import devin_shell
from charlie_work.devin_shell import launch_devin_session, read_session_records
from charlie_work.worker import iter_workers
from charlie_work.worktree import WorktreeInfo


def test_review_launch_uses_review_checkout_and_tears_down_on_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct unit test for ``launch_devin_session(review=True)`` (issue #1536).

    Until this test existed, the review-checkout branch was only verified by
    reading: ``test_devin_shell_template.py`` covers the command-template
    sanitizer and ``test_reviewer_adapter_routing.py`` mocks
    ``launch_devin_session`` away entirely. Here the real function runs with
    ``create_review_checkout``/``remove_review_checkout``/``popen_worker``
    (the adapter's Popen seam) patched, asserting:

    - the checkout call receives ``repo_root``, the PR number, ``head_sha``,
      and ``reviews_dir`` (the caller's ``sessions_dir``) -- never
      ``create_worktree``;
    - the review sidecar written under ``reviews_dir`` parses back through
      ``read_session_records`` and surfaces via ``iter_workers`` with
      ``adapter_kind == "devin"``;
    - a spawn ``OSError`` invokes ``remove_review_checkout`` teardown
      (never ``remove_worktree``) and still returns an error record rather
      than raising.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    prompt_path = tmp_path / "review-prompt.md"
    prompt_path.write_text("review the diff\n", encoding="utf-8")

    pr_number = 77
    branch = "agent/issue-7-fix"
    head_sha = "a" * 40

    checkout_calls: list[dict[str, Any]] = []
    teardown_calls: list[dict[str, Any]] = []

    def fake_create_review_checkout(
        repo_root_arg: Path,
        pr_number_arg: int,
        head_sha_arg: str,
        *,
        reviews_dir: Path,
    ) -> WorktreeInfo:
        checkout_calls.append(
            {
                "repo_root": repo_root_arg,
                "pr_number": pr_number_arg,
                "head_sha": head_sha_arg,
                "reviews_dir": reviews_dir,
            }
        )
        # Mirror the real path scheme (reviews_dir / "pr-<n>") so the launch
        # path has a real directory for cwd/env/marker writes.
        checkout_path = reviews_dir / f"pr-{pr_number_arg}"
        checkout_path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=checkout_path, branch=head_sha_arg, venv_junction=None)

    def fake_remove_review_checkout(
        repo_root_arg: Path, pr_number_arg: int, *, reviews_dir: Path
    ) -> bool:
        teardown_calls.append(
            {
                "repo_root": repo_root_arg,
                "pr_number": pr_number_arg,
                "reviews_dir": reviews_dir,
            }
        )
        return True

    def _forbid_worker_worktree_path(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "review=True must not touch the worker worktree path (create_worktree/remove_worktree)"
        )

    monkeypatch.setattr(devin_shell, "create_worktree", _forbid_worker_worktree_path)
    monkeypatch.setattr(devin_shell, "remove_worktree", _forbid_worker_worktree_path)
    monkeypatch.setattr(devin_shell, "create_review_checkout", fake_create_review_checkout)
    monkeypatch.setattr(devin_shell, "remove_review_checkout", fake_remove_review_checkout)

    popen_calls: list[dict[str, Any]] = []
    spawn_failure = {"enabled": False}

    def fake_popen_worker(args: Any, **kwargs: Any) -> Any:
        popen_calls.append({"argv": list(args), "cwd": kwargs.get("cwd")})
        if spawn_failure["enabled"]:
            raise OSError("simulated devin CLI spawn failure")
        return SimpleNamespace(pid=424242)

    monkeypatch.setattr(devin_shell, "popen_worker", fake_popen_worker)

    record = launch_devin_session(
        pr_number,
        branch,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        review=True,
        head_sha=head_sha,
    )

    assert record.error is None
    assert record.pid == 424242
    assert checkout_calls == [
        {
            "repo_root": repo_root,
            "pr_number": pr_number,
            "head_sha": head_sha,
            "reviews_dir": reviews_dir,
        }
    ]
    assert record.worktree_path == str(reviews_dir / f"pr-{pr_number}")
    assert popen_calls[0]["cwd"] == str(reviews_dir / f"pr-{pr_number}")
    # No teardown on the success path.
    assert teardown_calls == []

    # The review-mode sanitizer is applied regardless of the caller's
    # template: the worker default's "--permission-mode dangerous" must be
    # stripped from the rendered argv.
    assert "--permission-mode" not in record.command
    assert "dangerous" not in record.command

    # The review sidecar is a devin-adapter session record under reviews_dir:
    # readable by read_session_records and surfaced by iter_workers with
    # adapter_kind == "devin" (HARNESS_REGISTRY["devin-shell"].adapter_kind),
    # which is what REVIEWER_ADAPTER_KINDS-filtered reap logic keys on.
    sidecar_path = reviews_dir / f"issue-{pr_number}.json"
    assert sidecar_path.is_file()
    (persisted,) = read_session_records(reviews_dir)
    assert persisted.issue_number == pr_number
    assert persisted.pid == record.pid
    (view,) = iter_workers(reviews_dir)
    assert view.adapter_kind == "devin"
    assert view.issue_number == pr_number

    # Failure path: a spawn OSError must tear the review checkout down via
    # remove_review_checkout (never remove_worktree -- forbidden above) and
    # still come back as an error record, not an exception.
    spawn_failure["enabled"] = True

    failed = launch_devin_session(
        pr_number,
        branch,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        review=True,
        head_sha=head_sha,
    )

    assert failed.pid is None
    assert failed.error is not None
    assert "failed to launch devin" in failed.error
    assert len(checkout_calls) == 2
    assert teardown_calls == [
        {
            "repo_root": repo_root,
            "pr_number": pr_number,
            "reviews_dir": reviews_dir,
        }
    ]

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert "failed to launch devin" in payload["error"]
