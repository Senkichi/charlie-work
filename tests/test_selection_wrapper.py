"""``selection_wrapper``: the worker loop's ``ci-fleet test`` wiring.

charlie-work never selects tests itself; it wraps the repo's runner in ``ci-fleet
test``. These tests pin how the wrapper's executable, slug and base are derived, the
fail-soft fallbacks and their events, the prompt values in both modes, the
execution-contract guard, and the kill switch's path into worker environments. The
executable is always a stand-in file: ``tests/conftest.py`` keeps every other test
on today's commands.
"""

from __future__ import annotations

import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from _prompt_sections_fixtures import ISSUE_VALUES
from charlie_work import selection_wrapper
from charlie_work.config import OrchestratorConfig
from charlie_work.env_sanitize import sanitize_env
from charlie_work.paths import runtime_paths
from charlie_work.prompt_test_command import (
    IMPACTED_TESTS_PLACEHOLDER,
    PYTEST_FLAGS,
    SELECTED_STEP,
    TARGETED_STEP,
    UNRESOLVED_TARGETED,
    prompt_test_command_values,
)
from charlie_work.prompts import (
    EXECUTION_CONTRACT_MARKERS,
    SELECTION_CONTRACT_MARKERS,
    MissingExecutionContractError,
    assert_execution_contract,
    render_prompt,
)
from charlie_work.rework_prompts import _render_rework_prompt
from charlie_work.selection_wrapper import (
    DISABLED,
    ENFORCED,
    NO_BASE,
    NO_EXECUTABLE,
    NO_REPO,
    SHADOW,
    SelectionTarget,
    SelectionUnavailable,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.workflow import OrchestratorApp

# Captured at import, before tests/conftest.py's autouse stub replaces it.
_REAL_EXECUTABLE = selection_wrapper.ci_fleet_executable

RUNNER = "uv run --extra dev pytest"
UNAVAILABLE = "worker_test_selection_unavailable"
DISABLED_KIND = "worker_test_selection_disabled"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return done.stdout.strip()


def _repo(path: Path, *, origin: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    if origin is not None:
        _git(path, "remote", "add", "origin", origin)
    return path


def _state_file(tmp_path: Path) -> Path:
    return runtime_paths(tmp_path / "rt", OrchestratorConfig().runtime.state_dir).state_file


def _events(state_file: Path, kind: str) -> list[tuple[str, str, dict]]:
    db = state_file.parent / "events.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT kind, level, payload FROM events WHERE kind = ?", (kind,)
        ).fetchall()
    finally:
        conn.close()
    return [(k, level, json.loads(payload)) for k, level, payload in rows]


@pytest.fixture
def fake_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "venv" / "Scripts" / selection_wrapper.CI_FLEET_EXE_NAME
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(selection_wrapper, "ci_fleet_executable", lambda: exe)
    return exe


def _target(exe: Path = Path("/venv/bin/ci-fleet")) -> SelectionTarget:
    return SelectionTarget(exe=exe, repo="owner/repo-a", base="origin/main")


# ---------------------------------------------------------------------------
# The executable, the kill switch, enabled = false
# ---------------------------------------------------------------------------


def test_executable_is_the_console_script_beside_the_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert _REAL_EXECUTABLE() is None
    exe = tmp_path / selection_wrapper.CI_FLEET_EXE_NAME
    exe.write_text("", encoding="utf-8")
    assert _REAL_EXECUTABLE() == exe


def test_the_kill_switch_reaches_worker_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CI_FLEET_SELECT=off`` is honoured by the wrapper, so it must survive sanitizing."""
    monkeypatch.setenv("CI_FLEET_SELECT", "off")
    assert sanitize_env(tmp_path)["CI_FLEET_SELECT"] == "off"


@pytest.mark.parametrize(
    ("text", "disabled"),
    [
        (None, False),
        ("[test_selection]\nenabled = true\n", False),
        ("[test_selection]\nmin_level = 1\n", False),
        ("[test_selection]\nenabled = false\n", True),
        ("[test_selection\nenabled = false\n", False),
        ('[test_selection]\nenabled = "false"\n', False),
        ("[test_selection]\nenabled = 0\n", False),
    ],
)
def test_only_an_explicit_enabled_false_disables(
    tmp_path: Path, text: str | None, disabled: bool
) -> None:
    """A missing or malformed file is ci-fleet's to judge (it runs full and says why)."""
    if text is not None:
        (tmp_path / "ci-fleet.toml").write_text(text, encoding="utf-8")
    assert selection_wrapper.selection_disabled(tmp_path) is disabled


# ---------------------------------------------------------------------------
# Slug and base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "slug"),
    [
        ("https://github.com/owner/repo-a.git", "owner/repo-a"),
        ("https://github.com/owner/repo-a", "owner/repo-a"),
        ("git@github.com:owner/repo-a.git", "owner/repo-a"),
        ("https://gitlab.example/owner/repo-a.git", None),
    ],
)
def test_slug_comes_from_a_github_origin(tmp_path: Path, origin: str, slug: str | None) -> None:
    assert selection_wrapper.repo_slug(_repo(tmp_path / "r", origin=origin)) == slug


def test_a_repo_without_origin_is_local_by_directory_name(tmp_path: Path) -> None:
    assert selection_wrapper.repo_slug(_repo(tmp_path / "widget")) == "local/widget"


def test_configured_base_ref_wins(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    assert selection_wrapper.selection_base(repo, " origin/release ") == "origin/release"


def test_no_origin_bases_on_the_main_worktrees_branch(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    _git(repo, "checkout", "-q", "-b", "trunk")
    assert selection_wrapper.selection_base(repo, "") == "trunk"


def test_a_clone_bases_on_origins_default_branch(tmp_path: Path) -> None:
    upstream = _repo(tmp_path / "up")
    _git(upstream, "commit", "-q", "--allow-empty", "-m", "c")
    _git(tmp_path, "clone", "-q", str(upstream), "clone")
    assert selection_wrapper.selection_base(tmp_path / "clone", "") == "origin/main"


# ---------------------------------------------------------------------------
# resolve_selection and worker_selection
# ---------------------------------------------------------------------------


def test_missing_executable_is_unavailable(tmp_path: Path) -> None:
    result = selection_wrapper.resolve_selection(_repo(tmp_path / "r"))
    assert isinstance(result, SelectionUnavailable)
    assert result.reason == NO_EXECUTABLE


def test_no_repo_root_is_unavailable() -> None:
    assert selection_wrapper.resolve_selection(None).reason == NO_REPO


def test_disabled_repo_is_unavailable(tmp_path: Path, fake_exe: Path) -> None:
    repo = _repo(tmp_path / "r")
    (repo / "ci-fleet.toml").write_text("[test_selection]\nenabled = false\n", encoding="utf-8")
    assert selection_wrapper.resolve_selection(repo).reason == DISABLED


def test_non_github_origin_is_unavailable(tmp_path: Path, fake_exe: Path) -> None:
    repo = _repo(tmp_path / "r", origin="https://gitlab.example/o/r.git")
    assert selection_wrapper.resolve_selection(repo).reason == NO_REPO


def test_detached_local_repo_has_no_base(tmp_path: Path, fake_exe: Path) -> None:
    repo = _repo(tmp_path / "r")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "c")
    _git(repo, "checkout", "-q", "--detach")
    assert selection_wrapper.resolve_selection(repo).reason == NO_BASE


def test_a_resolvable_repo_yields_a_target(tmp_path: Path, fake_exe: Path) -> None:
    repo = _repo(tmp_path / "widget")
    assert selection_wrapper.resolve_selection(repo) == SelectionTarget(
        fake_exe, "local/widget", "main"
    )
    assert selection_wrapper.resolve_selection(repo, base="abc123").base == "abc123"


def test_worker_selection_returns_the_target_without_an_event(
    tmp_path: Path, fake_exe: Path
) -> None:
    state_file = _state_file(tmp_path)
    repo = _repo(tmp_path / "widget")
    target = selection_wrapper.worker_selection(
        repo, "", state_file=state_file, payload={"issue_number": 7}
    )
    assert target == SelectionTarget(fake_exe, "local/widget", "main")
    assert _events(state_file, UNAVAILABLE) == []


def test_a_missing_executable_is_a_warning_event(tmp_path: Path) -> None:
    state_file = _state_file(tmp_path)
    repo = _repo(tmp_path / "widget")
    assert (
        selection_wrapper.worker_selection(
            repo, "", state_file=state_file, payload={"issue_number": 7}
        )
        is None
    )
    [(kind, level, payload)] = _events(state_file, UNAVAILABLE)
    assert (kind, level) == (UNAVAILABLE, "warning")
    assert (payload["issue_number"], payload["reason"]) == (7, NO_EXECUTABLE)


def test_a_disabled_repo_is_an_info_event(tmp_path: Path, fake_exe: Path) -> None:
    state_file = _state_file(tmp_path)
    repo = _repo(tmp_path / "widget")
    (repo / "ci-fleet.toml").write_text("[test_selection]\nenabled = false\n", encoding="utf-8")
    assert selection_wrapper.worker_selection(repo, "", state_file=state_file, payload={}) is None
    assert [(k, lvl) for k, lvl, _ in _events(state_file, DISABLED_KIND)] == [
        (DISABLED_KIND, "info")
    ]
    assert _events(state_file, UNAVAILABLE) == []


def test_no_repo_root_is_silent(tmp_path: Path) -> None:
    """Only test callers render without a repo root; production always passes one."""
    state_file = _state_file(tmp_path)
    assert selection_wrapper.worker_selection(None, "", state_file=state_file, payload={}) is None
    assert _events(state_file, UNAVAILABLE) == []


# ---------------------------------------------------------------------------
# Command shapes and the shadow-window probe
# ---------------------------------------------------------------------------


def test_worker_command_wraps_the_runner_and_splits_back_exactly() -> None:
    target = _target()
    command = selection_wrapper.worker_command(target, RUNNER, PYTEST_FLAGS)
    assert shlex.split(command) == [
        target.exe.as_posix(),
        "test",
        "--repo",
        "owner/repo-a",
        "--base",
        "origin/main",
        "--context",
        "worker",
        "--",
        *shlex.split(RUNNER),
        "-q",
        "--tb=short",
    ]


def test_worker_command_quotes_an_executable_path_with_spaces() -> None:
    target = _target(Path("/Program Files/ci-fleet"))
    command = selection_wrapper.worker_command(target, RUNNER, PYTEST_FLAGS)
    assert shlex.split(command)[0] == target.exe.as_posix()


@pytest.mark.parametrize("shadow", [True, False])
def test_gate_argv_wraps_the_suite_with_head_and_base(shadow: bool) -> None:
    target = _target()
    argv = selection_wrapper.gate_argv(target, ["pytest", "-q"], head="h" * 40, shadow=shadow)
    assert argv == [
        str(target.exe),
        "test",
        "--repo",
        "owner/repo-a",
        "--base",
        "origin/main",
        "--head",
        "h" * 40,
        "--context",
        "gate",
        *(["--shadow"] if shadow else []),
        "--",
        "pytest",
        "-q",
    ]


@pytest.mark.parametrize(
    ("result", "met"),
    [
        (RunResult(returncode=0, stdout="{}", stderr=""), True),
        (RunResult(returncode=1, stdout="{}", stderr=""), False),
        (RunResult(returncode=None, stdout="", stderr="", timed_out=True), False),
        (RunResult(returncode=None, stdout="", stderr="", error="not found"), False),
    ],
)
def test_shadow_window_is_shadow_status_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: RunResult, met: bool
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> RunResult:
        calls.append(command)
        return result

    monkeypatch.setattr(selection_wrapper, "run_captured", fake_run)
    exe = Path("/venv/bin/ci-fleet")
    assert selection_wrapper.shadow_window_met(exe, "owner/repo-a", cwd=tmp_path) is met
    assert calls == [[str(exe), "nightly", "shadow-status", "--repo", "owner/repo-a"]]


def test_gate_selection_without_executable_keeps_the_argv(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "widget")
    selection = selection_wrapper.gate_selection(repo, ["pytest"], head="h", base="b")
    assert (selection.argv, selection.mode) == (("pytest",), NO_EXECUTABLE)


def test_gate_selection_without_head_or_base_keeps_the_argv(
    tmp_path: Path, fake_exe: Path
) -> None:
    repo = _repo(tmp_path / "widget")
    for head, base in ((None, "b"), ("h", None)):
        selection = selection_wrapper.gate_selection(repo, ["pytest"], head=head, base=base)
        assert (selection.argv, selection.mode) == (("pytest",), NO_BASE)


@pytest.mark.parametrize(("met", "mode"), [(False, SHADOW), (True, ENFORCED)])
def test_gate_selection_follows_the_shadow_window(
    tmp_path: Path, fake_exe: Path, monkeypatch: pytest.MonkeyPatch, met: bool, mode: str
) -> None:
    repo = _repo(tmp_path / "widget")
    asked: list[str] = []

    def window(exe: Path, slug: str, *, cwd: Path) -> bool:
        asked.append(slug)
        return met

    monkeypatch.setattr(selection_wrapper, "shadow_window_met", window)
    selection = selection_wrapper.gate_selection(
        repo, ["pytest", "-q"], head="h" * 40, base="b" * 40
    )
    assert (selection.mode, asked) == (mode, ["local/widget"])
    target = SelectionTarget(fake_exe, "local/widget", "b" * 40)
    expected = selection_wrapper.gate_argv(target, ["pytest", "-q"], head="h" * 40, shadow=not met)
    assert selection.argv == tuple(expected)


# ---------------------------------------------------------------------------
# Prompt values and the execution-contract guard
# ---------------------------------------------------------------------------

SELECTED_VALUES = prompt_test_command_values(RUNNER, None, selection=_target())


def test_values_without_selection_are_the_targeted_form() -> None:
    values = prompt_test_command_values(RUNNER, None)
    assert (
        values["targeted_test_command"] == f"{RUNNER} {IMPACTED_TESTS_PLACEHOLDER} {PYTEST_FLAGS}"
    )
    assert values["test_step_instruction"] == TARGETED_STEP
    assert all(m in values["test_execution_contract"] for m in EXECUTION_CONTRACT_MARKERS)
    assert f"`{RUNNER} {PYTEST_FLAGS}`" in values["test_execution_contract"]


def test_values_with_selection_run_ci_fleet_test() -> None:
    assert SELECTED_VALUES["targeted_test_command"] == selection_wrapper.worker_command(
        _target(), RUNNER, PYTEST_FLAGS
    )
    assert SELECTED_VALUES["test_step_instruction"] == SELECTED_STEP
    assert "grep tests/" not in SELECTED_STEP
    assert "`--also <path> ...` before the `--`" in SELECTED_STEP
    contract = SELECTED_VALUES["test_execution_contract"]
    assert all(m in contract for m in SELECTION_CONTRACT_MARKERS)
    assert not any(m in contract for m in EXECUTION_CONTRACT_MARKERS)
    assert "CI runs the same selection or the full suite, and the nightly runs everything" in (
        contract
    )
    assert f"`{RUNNER} {PYTEST_FLAGS}`" in contract


def test_selection_without_a_runner_stays_unresolved() -> None:
    values = prompt_test_command_values("", None, selection=_target())
    assert values["targeted_test_command"] == UNRESOLVED_TARGETED
    assert values["test_step_instruction"] == TARGETED_STEP


@pytest.mark.parametrize("template", ["worker.md", "worker_claude_code.md", "rework.md"])
def test_selected_prompts_pass_the_execution_contract_guard(template: str) -> None:
    prompt = render_prompt(template, {**ISSUE_VALUES, **SELECTED_VALUES})
    assert_execution_contract(prompt)
    assert SELECTED_VALUES["targeted_test_command"] in prompt
    assert SELECTED_STEP in prompt
    assert "grep tests/" not in prompt


def test_the_guard_needs_one_marker_set_in_full() -> None:
    with pytest.raises(MissingExecutionContractError):
        assert_execution_contract(EXECUTION_CONTRACT_MARKERS[0] + SELECTION_CONTRACT_MARKERS[1])
    with pytest.raises(MissingExecutionContractError):
        assert_execution_contract(SELECTION_CONTRACT_MARKERS[0] + EXECUTION_CONTRACT_MARKERS[1])
    assert_execution_contract("".join(SELECTION_CONTRACT_MARKERS))
    assert_execution_contract("".join(EXECUTION_CONTRACT_MARKERS))


# ---------------------------------------------------------------------------
# Wiring: the real worker writer and rework renderer
# ---------------------------------------------------------------------------


def _app(tmp_path: Path) -> OrchestratorApp:
    repo = _repo(tmp_path / "widget")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "w"\nversion = "0"\n'
        '[project.optional-dependencies]\ndev = ["pytest"]\n',
        encoding="utf-8",
    )
    config = OrchestratorConfig()
    return OrchestratorApp(repo, runtime_paths(repo, config.runtime.state_dir), config, gh=None)


ISSUE = {"number": 1, "title": "T", "url": "u", "body": "b"}


def test_the_worker_writer_renders_the_selected_command(tmp_path: Path, fake_exe: Path) -> None:
    app = _app(tmp_path)
    text = app._write_worker_prompt(ISSUE).read_text(encoding="utf-8")
    target = SelectionTarget(fake_exe, "local/widget", "main")
    assert selection_wrapper.worker_command(target, RUNNER, PYTEST_FLAGS) in text
    assert SELECTION_CONTRACT_MARKERS[0] in text


def test_the_worker_writer_falls_back_and_records_why(tmp_path: Path) -> None:
    app = _app(tmp_path)
    text = app._write_worker_prompt(ISSUE).read_text(encoding="utf-8")
    assert f"{RUNNER} {IMPACTED_TESTS_PLACEHOLDER} {PYTEST_FLAGS}" in text
    assert EXECUTION_CONTRACT_MARKERS[0] in text
    [(_, _, payload)] = _events(app.paths.state_file, UNAVAILABLE)
    assert (payload["issue_number"], payload["reason"]) == (1, NO_EXECUTABLE)


def test_the_rework_renderer_renders_the_selected_command(tmp_path: Path, fake_exe: Path) -> None:
    app = _app(tmp_path)
    pr = {"number": 5, "title": "t", "url": "u", "headRefName": "agent/issue-1-t"}
    text = _render_rework_prompt(
        app.paths.state_file, pr, 1, "A note.", app.config, repo_root=app.repo_root
    )
    target = SelectionTarget(fake_exe, "local/widget", "main")
    assert selection_wrapper.worker_command(target, RUNNER, PYTEST_FLAGS) in text
    assert SELECTION_CONTRACT_MARKERS[0] in text
