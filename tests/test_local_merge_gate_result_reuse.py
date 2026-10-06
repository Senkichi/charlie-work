"""Issue #2125: the local merge gate reuses a passing suite result for the same
(head, base) pair instead of rerunning a 25-35 minute suite.

Unit tests pin ``reusable_gate_result``'s fail-closed contract; the integration
tests drive the real gate through ``_local_merge_approved`` with a real git repo
and real runner processes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys

from charlie_work import local_suite_runner, selection_wrapper
from charlie_work.local_lane import branch_head_sha, is_ancestor, suite_command_argv
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _local_gate_async_fixtures import (  # noqa: E402
    FAIL_SUITE,
    _adopt_and_approve,
    _commit_file,
    _event_kinds,
    _events_of_kind,
    _gate_paths,
    _init_repo,
    _kill_claimed_gate,
    _wait_pid_dead,
    _lane_app,
    _lane_config,
    _make_branch,
    _wait_for_result,
)
from charlie_work.orchestration import local_merge_gate  # noqa: E402,F401

HEAD = "a" * 40
BASE = "b" * 40
ARGV = ["uv", "run", "pytest", "-q", "--tb=short"]


def _good_result(**overrides: object) -> dict:
    return {
        "ok": True,
        "returncode": 0,
        "head_sha": HEAD,
        "base_sha": BASE,
        "suite_argv": ARGV,
        "ended_at": "2026-01-01T00:00:00Z",
        **overrides,
    }


def _paths(tmp_path: Path) -> local_suite_runner.SuiteGatePaths:
    paths = local_suite_runner.suite_gate_paths(tmp_path, 7)
    paths.gate_dir.mkdir(parents=True)
    return paths


def _reusable(paths: local_suite_runner.SuiteGatePaths, head=HEAD, base=BASE, argv=ARGV):
    return local_suite_runner.reusable_gate_result(
        paths, head_sha=head, base_sha=base, suite_argv=argv
    )


def test_same_pair_ok_result_is_reusable(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.result.write_text(json.dumps(_good_result()), encoding="utf-8")
    assert _reusable(paths) is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_sha": "c" * 40},
        {"head_sha": "c" * 40},
        {"ok": False, "returncode": 1},
        {"ok": "true"},
        {"ok": 1},
        {"ended_at": None},
        {"base_sha": None},
        {"suite_argv": ["uv", "run", "pytest"]},
        {"suite_argv": ["ci-fleet", "test", "--context", "gate", "--", *ARGV]},
        {"suite_argv": None},
    ],
)
def test_mismatched_or_unfinished_result_is_a_miss(tmp_path: Path, overrides: dict) -> None:
    paths = _paths(tmp_path)
    paths.result.write_text(json.dumps(_good_result(**overrides)), encoding="utf-8")
    assert _reusable(paths) is None


@pytest.mark.parametrize("key", ["ended_at", "suite_argv"])
def test_missing_key_is_a_miss(tmp_path: Path, key: str) -> None:
    paths = _paths(tmp_path)
    payload = _good_result()
    del payload[key]
    paths.result.write_text(json.dumps(payload), encoding="utf-8")
    assert _reusable(paths) is None


@pytest.mark.parametrize("raw", ["", '{"ok": true, "head_sha": "', "[1, 2]", "not json"])
def test_truncated_or_malformed_result_is_a_miss(tmp_path: Path, raw: str) -> None:
    paths = _paths(tmp_path)
    paths.result.write_text(raw, encoding="utf-8")
    assert _reusable(paths) is None


def test_absent_result_and_empty_shas_are_a_miss(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert _reusable(paths) is None
    paths.result.write_text(json.dumps(_good_result(head_sha="", base_sha="")), encoding="utf-8")
    assert _reusable(paths, head="", base="") is None
    assert _reusable(paths, head=HEAD, base=None) is None


# ---------------------------------------------------------------------------
# Integration: the real gate
# ---------------------------------------------------------------------------


@pytest.fixture
def lane_repo() -> Path:
    return Path(tempfile.mkdtemp(prefix="cw-gate-reuse-"))


def _drop_claim(app: OrchestratorApp, pr_number: int) -> None:
    """Lose the claim (supervisor restart / re-arm) while the result file stays."""
    # The result lands just before the runner exits; a still-live pid file would
    # make the gate defer instead of launching.
    _wait_pid_dead(
        int(load_state_locked(app.paths.state_file)["prs"][str(pr_number)]["local_suite_pid"])
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        record = state["prs"][str(pr_number)]
        for field in local_merge_gate.LOCAL_SUITE_CLAIM_FIELDS:
            record.pop(field, None)
        save_state(app.paths.state_file, state)


def test_same_head_and_base_reuses_result_without_launching(lane_repo: Path) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    first = _wait_for_result(app, 7)
    assert first["ok"] is True
    _drop_claim(app, 7)

    results = app._local_merge_approved()

    assert results[0]["outcome"] in ("merged", "already_merged")
    assert results[0]["reused_result"] is True
    assert is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
    assert len(_events_of_kind(app, "local_suite_launched")) == 1
    reused = _events_of_kind(app, "local_merge_gate_result_reused")
    assert len(reused) == 1
    assert reused[0]["payload"]["head_sha"] == first["head_sha"]
    assert reused[0]["payload"]["base_sha"] == first["base_sha"]
    assert load_state_locked(app.paths.state_file)["prs"]["7"]["status"] == "merged"


def test_moved_base_reruns_the_suite(lane_repo: Path) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    _wait_for_result(app, 7)
    _drop_claim(app, 7)
    _commit_file(lane_repo, "other.py", "x = 1\n", "another PR merged first")

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        assert "local_merge_gate_result_reused" not in _event_kinds(app)
        assert len(_events_of_kind(app, "local_suite_launched")) == 2
    finally:
        _kill_claimed_gate(app, 7)


def test_failed_prior_result_reruns_the_suite(lane_repo: Path) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": FAIL_SUITE})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    assert _wait_for_result(app, 7)["ok"] is False
    _drop_claim(app, 7)

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        assert "local_merge_gate_result_reused" not in _event_kinds(app)
    finally:
        _kill_claimed_gate(app, 7)


def test_truncated_prior_result_reruns_and_is_never_a_pass(lane_repo: Path) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    _wait_for_result(app, 7)
    _drop_claim(app, 7)
    _gate_paths(app, 7).result.write_text('{"ok": true, "head_sha": "', encoding="utf-8")

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        assert "local_merge_gate_result_reused" not in _event_kinds(app)
        assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
    finally:
        _kill_claimed_gate(app, 7)


# ---------------------------------------------------------------------------
# Test impact selection: the gate's suite runs inside ``ci-fleet test``
# ---------------------------------------------------------------------------


def _use_selection(monkeypatch: pytest.MonkeyPatch, *, window_met: bool) -> None:
    """The interpreter stands in for ``ci-fleet``; the shadow window is fixed.

    The launched ``python test --repo ...`` exits at once (no file ``test``),
    which is all these tests need: they read the launch record, not the run.
    """
    monkeypatch.setattr(selection_wrapper, "ci_fleet_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(
        selection_wrapper, "shadow_window_met", lambda exe, repo, *, cwd: window_met
    )


@pytest.mark.parametrize(("window_met", "mode"), [(False, "shadow"), (True, "enforced")])
def test_the_gate_runs_its_suite_inside_ci_fleet_test(
    lane_repo: Path, monkeypatch: pytest.MonkeyPatch, window_met: bool, mode: str
) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    _use_selection(monkeypatch, window_met=window_met)
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    try:
        assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
        [launched] = _events_of_kind(app, "local_suite_launched")
        record = load_state_locked(app.paths.state_file)["prs"]["7"]
        argv = launched["payload"]["argv"]
        split = argv.index("--")
        assert argv[:4] == [sys.executable, "test", "--repo", f"local/{lane_repo.name}"]
        assert argv[argv.index("--base") + 1] == record["local_suite_base_sha"]
        assert argv[argv.index("--head") + 1] == record["local_suite_head"]
        assert argv[argv.index("--context") + 1] == "gate"
        assert ("--shadow" in argv[:split]) is (not window_met)
        assert argv[split + 1 :] == suite_command_argv(app.config.dispatch.test_command, lane_repo)
        assert launched["payload"]["selection"] == mode
        assert record["local_suite_argv"] == argv
    finally:
        _kill_claimed_gate(app, 7)


def test_without_ci_fleet_the_gate_runs_unwrapped_and_says_why(lane_repo: Path) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    try:
        assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
        [launched] = _events_of_kind(app, "local_suite_launched")
        expected = suite_command_argv(app.config.dispatch.test_command, lane_repo)
        assert launched["payload"]["argv"] == expected
        assert launched["payload"]["selection"] == selection_wrapper.NO_EXECUTABLE
        [fallback] = _events_of_kind(app, "local_suite_selection_unavailable")
        assert fallback["payload"]["reason"] == selection_wrapper.NO_EXECUTABLE
        assert fallback["payload"]["pr_number"] == 7
    finally:
        _kill_claimed_gate(app, 7)


def test_a_disabled_repo_runs_unwrapped_with_an_info_event(lane_repo: Path, monkeypatch) -> None:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    (lane_repo / "ci-fleet.toml").write_text(
        "[test_selection]\nenabled = false\n", encoding="utf-8"
    )
    _use_selection(monkeypatch, window_met=True)
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    try:
        assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
        [launched] = _events_of_kind(app, "local_suite_launched")
        assert launched["payload"]["selection"] == selection_wrapper.DISABLED
        assert launched["payload"]["argv"][0] != sys.executable
        assert len(_events_of_kind(app, "local_suite_selection_disabled")) == 1
        assert _events_of_kind(app, "local_suite_selection_unavailable") == []
    finally:
        _kill_claimed_gate(app, 7)


def test_turning_selection_on_reruns_instead_of_reusing(
    lane_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass of the unwrapped suite is not a pass of the wrapped one: same pair, new argv."""
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)

    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    assert _wait_for_result(app, 7)["ok"] is True
    _drop_claim(app, 7)
    _use_selection(monkeypatch, window_met=False)

    results = app._local_merge_approved()

    try:
        assert results[0]["outcome"] == "suite_launched"
        assert "local_merge_gate_result_reused" not in _event_kinds(app)
        launches = _events_of_kind(app, "local_suite_launched")
        assert [e["payload"]["selection"] for e in launches] == ["no-executable", "shadow"]
    finally:
        _kill_claimed_gate(app, 7)
