"""Review-dispatch launches: ``launch_claude_worker(review=True)``
isolated checkout, permission mode, model/effort/max-turns pinning,
and terminal status records.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _init_real_repo,
    _repo_head_sha,
)

from charlie_work import claude_code
from charlie_work.config import (
    ClaudeCodeConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    ReviewerRoleConfig,
)
from charlie_work.claude_code import (
    launch_claude_worker,
    _review_effort_arm,
)

# ---------------------------------------------------------------------------
# Review-dispatch isolation (issue #370/#397): launch_claude_worker(review=True)
# must route through create_review_checkout (a PR-keyed, detached-HEAD
# checkout), never create_worktree (the worker's branch-slug worktree). These
# tests exercise the REAL worktree.create_review_checkout/create_worktree
# code, not a monkeypatched stand-in, so a regression that re-routes review
# through the shared worker worktree would fail them.
# ---------------------------------------------------------------------------


def test_launch_claude_worker_review_uses_isolated_checkout_not_worker_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A review=True launch must never call create_worktree — only
    create_review_checkout — and must land in a directory distinct from a
    live worker's worktree for the very same branch.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    branch = "agent/issue-500-fix"
    head_sha = _repo_head_sha(repo_root)

    # Simulate a live worker worktree for this branch, using the real
    # (non-monkeypatched) create_worktree, sitting under a separate dir.
    from charlie_work.worktree import create_worktree

    worker_info = create_worktree(
        repo_root, branch, base_ref="HEAD", worktrees_dir=tmp_path / "worktrees"
    )
    worker_marker = worker_info.path / "worker-in-progress.txt"
    worker_marker.write_text("do not touch\n", encoding="utf-8")

    def _forbid_create_worktree(*_args, **_kwargs):
        raise AssertionError(
            "review=True must never call create_worktree — it must use "
            "create_review_checkout instead"
        )

    monkeypatch.setattr(claude_code, "create_worktree", _forbid_create_worktree)

    record = launch_claude_worker(
        500,
        branch,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
    )

    assert record.ok, record.error
    review_path = Path(record.worktree_path)
    assert review_path != worker_info.path
    assert review_path.parent == sessions_dir

    # The worker's worktree is completely untouched by the review launch.
    assert worker_info.path.exists()
    assert worker_marker.exists()
    assert worker_marker.read_text(encoding="utf-8") == "do not touch\n"


def test_launch_claude_worker_review_defaults_to_read_only_permission_mode(
    tmp_path: Path,
) -> None:
    """Reviewer sessions default to --permission-mode plan (read-only), not
    the worker default of acceptEdits."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        501,
        "agent/issue-501-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
    )

    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "plan"
    assert "acceptEdits" not in record.command


def test_launch_claude_worker_review_ignores_caller_command_template_override(
    tmp_path: Path,
) -> None:
    """Round-2 review (PR #397): a reviewer's read-only posture must not be
    defeatable by an operator's worker-tuning `claude_code.command` override.

    workflow.dispatch_reviews forwards `command_template` from
    ClaudeCodeConfig.command only when non-empty (see workflow.py); this test
    simulates the exact defeat scenario the round-2 verdict describes — an
    operator who has uncommented the example config's acceptEdits override
    for worker-tuning reasons — by passing a non-empty, acceptEdits-bearing
    command_template straight into launch_claude_worker(review=True, ...).
    The reviewer must still launch with plan mode: the adapter hard-pins the
    review command template and ignores any caller-supplied override.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    operator_worker_override = ("claude", "-p", "--permission-mode", "acceptEdits")

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        command_template=operator_worker_override,
    )

    # Matches the sibling default-permission-mode test's convention: assert
    # on the rendered argv, not on launch success (the `claude` binary need
    # not be spawnable in every test environment — the command is rendered
    # and recorded before the process-launch step either way).
    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "plan"
    assert "acceptEdits" not in record.command


def test_launch_claude_worker_review_pins_configured_model_by_default(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        502,
        "agent/issue-502-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
    )

    assert "--model" in record.command
    idx = record.command.index("--model")
    assert record.command[idx + 1] == claude_code._DEFAULT_CLAUDE_MODEL


def test_launch_claude_worker_review_pins_max_turns_override(tmp_path: Path) -> None:
    """Issue #1439 round-2 review: ``launch_claude_worker(review=True,
    max_turns_override=N)`` must construct a command that hard-pins
    ``--max-turns`` to N — not the flat ``review_max_turns`` config default.

    The #1439 dispatch-path tests monkeypatch ``launch_claude_worker`` itself
    and only assert on the ``max_turns_override`` kwarg passed through, so they
    never exercise the actual command-construction mechanism
    (``_apply_max_turns_pin``). This test drives the REAL
    ``launch_claude_worker`` (real ``create_review_checkout``, real
    ``subprocess.Popen`` of a fake claude script) and asserts on
    ``record.command`` — the fully-rendered argv handed to ``popen_worker`` —
    so a regression that drops, ignores, or mis-wires the override is caught.

    Uses 120 (distinct from the default 40) so a mutation that accepts the
    parameter but silently falls back to the config default also fails this
    test, not just one that removes the parameter.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        1439,
        "agent/issue-1439-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
        max_turns_override=120,
    )

    assert record.ok, record.error
    assert record.command.count("--max-turns") == 1
    idx = record.command.index("--max-turns")
    assert record.command[idx + 1] == "120"
    # Distinct from the flat config default — proves the override won, not a
    # coincidental match.
    assert record.command[idx + 1] != str(ReviewDispatchConfig().review_max_turns)


def test_launch_claude_worker_review_max_turns_override_none_uses_config_default(
    tmp_path: Path,
) -> None:
    """Issue #1439 round-2 review: ``launch_claude_worker(review=True,
    max_turns_override=None)`` must use
    ``resolved_config.review_dispatch.review_max_turns`` exactly as before —
    regression protection for every direct caller and unit test that does not
    opt into the structure-aware cap.

    Uses a non-default ``review_max_turns`` (25) so the assertion is
    meaningful: it confirms the value is read from config, not from a
    hardcoded constant that happens to match the default.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(review_max_turns=25),
    )

    record = launch_claude_worker(
        1440,
        "agent/issue-1440-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
        config=config,
        max_turns_override=None,
    )

    assert record.ok, record.error
    assert record.command.count("--max-turns") == 1
    idx = record.command.index("--max-turns")
    assert record.command[idx + 1] == "25"


def test_launch_claude_worker_review_uses_review_effort_when_set(tmp_path: Path) -> None:
    """A reviewer session must pin reviewer.effort over claude_code.effort
    when reviewer.effort is explicitly set."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        reviewer=ReviewerRoleConfig(effort="high"),
    )

    record = launch_claude_worker(
        503,
        "agent/issue-503-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "high"


def test_launch_claude_worker_review_falls_back_to_claude_code_effort_when_unset(
    tmp_path: Path,
) -> None:
    """reviewer.effort empty (the default) must fall back to
    claude_code.effort, same as any other reviewer launch before this config
    knob existed."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="medium"),
        reviewer=ReviewerRoleConfig(effort=""),
    )

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "medium"


def test_launch_claude_worker_review_experiment_treatment_pins_review_effort(
    tmp_path: Path,
) -> None:
    """Experiment enabled (fraction=1.0, so every PR is treatment) --> the
    reviewer session pins reviewer.effort, same as the always-on case."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        reviewer=ReviewerRoleConfig(effort="high", effort_experiment_fraction=1.0),
    )

    record = launch_claude_worker(
        601,
        "agent/issue-601-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "high"


def test_launch_claude_worker_review_experiment_control_falls_back(tmp_path: Path) -> None:
    """Experiment enabled with a vanishingly small fraction --> an arbitrary
    PR is (deterministically) assigned control and falls back to
    claude_code.effort instead of review_effort."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    pr_number = 602
    tiny_fraction = 1e-9
    assert _review_effort_arm(pr_number, tiny_fraction, "") is False
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        reviewer=ReviewerRoleConfig(effort="high", effort_experiment_fraction=tiny_fraction),
    )

    record = launch_claude_worker(
        pr_number,
        "agent/issue-602-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "low"


def test_launch_claude_worker_review_uses_resolved_review_effort_passthrough(
    tmp_path: Path,
) -> None:
    """When the caller (dispatch_reviews) already resolved the review_effort
    experiment arm at claim time and passes it via resolved_review_effort,
    launch_claude_worker must use that value directly rather than
    re-resolving from config -- this is the single-computation-site
    invariant: the claim-time resolution is authoritative, not a preview."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    # Config alone would resolve to "high" (fraction=1.0, reviewer.effort=high),
    # but the passed-through resolved_review_effort deliberately differs so
    # the test can distinguish "used the passthrough" from "recomputed".
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        reviewer=ReviewerRoleConfig(effort="high", effort_experiment_fraction=1.0),
    )

    record = launch_claude_worker(
        603,
        "agent/issue-603-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
        resolved_review_effort="medium",
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "medium"


def test_launch_claude_worker_review_missing_head_sha_returns_error_record(
    tmp_path: Path,
) -> None:
    """review=True without head_sha is a caller error (ValueError), surfaced
    as an error record — launch_claude_worker must never raise."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"

    record = launch_claude_worker(
        503,
        "agent/issue-503-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=None,
    )

    assert not record.ok
    assert record.pid is None
    assert "head_sha" in record.error


def test_launch_claude_worker_review_prompt_write_failure_tears_down_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If writing the prompt file fails in review mode, the isolated review
    checkout — not a worker worktree — is torn down (via
    remove_review_checkout, keyed by PR number, not remove_worktree)."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error

    checkout_path = sessions_dir / "pr-504"
    assert not checkout_path.exists()
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(checkout_path) not in result.stdout


def test_launch_claude_worker_review_writes_terminal_status_record(
    tmp_path: Path,
) -> None:
    """A review=True launch must start the terminal-status watcher so a
    ``terminal.json`` appears at
    ``worker_terminal_status_path(reviews_dir, pr_number, 'claude')`` once the
    reviewer process exits (issue #1354, PR #1356 round-2 review).

    Before the fix, ``launch_claude_worker`` guarded the
    ``start_terminal_status_watcher`` call with ``if not review:``, so review
    launches never wrote a terminal-status record and the review-verdict
    reaper's exit-code fallback had nothing to read. This test exercises the
    real launch path end-to-end (real ``create_review_checkout``, real
    ``subprocess.Popen`` of the fake claude script, real watcher thread) and
    asserts the durable record materializes at the path the reaper reads from.
    """
    from charlie_work.process_utils import (
        find_worker_terminal_status,
        worker_terminal_status_path,
    )

    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        1354,
        "agent/issue-1354-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
    )

    assert record.ok, record.error
    assert record.pid is not None

    expected_path = worker_terminal_status_path(sessions_dir, 1354, "claude")
    # The watcher polls every _TERMINAL_STATUS_POLL_INTERVAL_SECONDS (2s); the
    # fake script exits near-instantly, so the record should appear within a
    # few poll intervals. Poll rather than sleep a fixed duration so the test
    # is fast on a healthy path and only waits as long as needed.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not expected_path.exists():
        time.sleep(0.1)
    assert expected_path.exists(), (
        f"review=True launch did not write a terminal-status record at "
        f"{expected_path} -- the start_terminal_status_watcher guard "
        f"(`if not review:`) may have been reintroduced"
    )

    payload = find_worker_terminal_status(sessions_dir, 1354)
    assert payload is not None, "terminal-status record vanished after appearing"
    assert payload["pid"] == record.pid
    # The fake claude script exits 0.
    assert payload["exit_code"] == 0
