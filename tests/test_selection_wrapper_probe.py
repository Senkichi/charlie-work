"""``selection_wrapper.probe_selection``: don't tell a worker to run a selector that can only say full.

``ci-fleet test`` runs the whole suite when the repo's selection map is stale or
missing, and for a worker that is a serial run which outlasts its foreground limit.
``worker_selection`` therefore asks ``ci-fleet select`` (against an empty diff, where a
``full`` verdict holds for every change) before the prompt commits to the wrapper, and
falls back to the hand-picked targeted command when the answer is full. These tests pin
the probe's argv, the fallback and its single event, the fail-open table, the
``dispatch.worker_selection_probe`` kill switch through both prompt writers, the
corrected contract text, and that the merge gate is never probed. The executable is
always a stand-in file, like ``tests/test_selection_wrapper.py``.
"""

from __future__ import annotations

import json
import shlex
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from _prompt_sections_fixtures import ISSUE_VALUES
from charlie_work import selection_wrapper
from charlie_work.config import DispatchConfig, OrchestratorConfig, build_config_from_data
from charlie_work.instrumentation import correlation_context
from charlie_work.paths import runtime_paths
from charlie_work.prompt_test_command import _selected_contract, prompt_test_command_values
from charlie_work.prompts import (
    EXECUTION_CONTRACT_MARKERS,
    SELECTION_CONTRACT_MARKERS,
    assert_execution_contract,
    render_prompt,
)
from charlie_work.rework_prompts import _render_rework_prompt
from charlie_work.selection_wrapper import (
    SELECT_PROBE_TIMEOUT_SECONDS,
    SELECTION_FULL,
    SelectionTarget,
    SelectionUnavailable,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.workflow import OrchestratorApp

# Captured at import, before any test patches it: git calls must still run for real.
_REAL_RUN_CAPTURED = selection_wrapper.run_captured

RUNNER = "uv run --extra dev pytest"
UNAVAILABLE = "worker_test_selection_unavailable"
ISSUE = {"number": 1, "title": "T", "url": "u", "body": "b"}
PR = {"number": 5, "title": "t", "url": "u", "headRefName": "agent/issue-1-t"}

FULL_MAP_STALE = json.dumps({"mode": "full", "reason": "map-stale"})
SELECTED_EMPTY = json.dumps({"mode": "selected", "tests": []})


def _ok(stdout: str) -> RunResult:
    return RunResult(returncode=0, stdout=stdout, stderr="")


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


def _repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    return path


def _state_file(tmp_path: Path) -> Path:
    return runtime_paths(tmp_path / "rt", OrchestratorConfig().runtime.state_dir).state_file


def _events(state_file: Path, kind: str) -> list[dict]:
    db = state_file.parent / "events.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT payload FROM events WHERE kind = ?", (kind,)).fetchall()
    finally:
        conn.close()
    return [json.loads(payload) for (payload,) in rows]


@pytest.fixture
def fake_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "venv" / "Scripts" / selection_wrapper.CI_FLEET_EXE_NAME
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(selection_wrapper, "ci_fleet_executable", lambda: exe)
    return exe


@pytest.fixture
def widget(tmp_path: Path) -> Path:
    return _repo(tmp_path / "widget")


class ExeCalls:
    """Answers every ``ci-fleet`` subcommand from a script and records each call.

    Anything that is not the stand-in executable (the git calls that derive the slug
    and base) runs for real, so the resolved target is the one production would build.
    """

    def __init__(self, exe: Path, answers: dict[str, RunResult]) -> None:
        self.exe = str(exe)
        self.answers = answers
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command: list[str], **kwargs: object) -> RunResult:
        if command[0] != self.exe:
            return _REAL_RUN_CAPTURED(command, **kwargs)  # type: ignore[arg-type]
        self.calls.append((command, kwargs))
        return self.answers.get(
            command[1], RunResult(returncode=1, stdout="", stderr="", error="unscripted")
        )

    def subcommands(self) -> list[str]:
        return [argv[1] for argv, _ in self.calls]


@pytest.fixture
def answering(fake_exe: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[RunResult], ExeCalls]:
    """Install a ``run_captured`` that answers ``ci-fleet select`` with the given result."""

    def install(select_result: RunResult) -> ExeCalls:
        runner = ExeCalls(fake_exe, {"select": select_result})
        monkeypatch.setattr(selection_wrapper, "run_captured", runner)
        return runner

    return install


def _worker_selection(widget: Path, state_file: Path, **kwargs: bool) -> SelectionTarget | None:
    return selection_wrapper.worker_selection(
        widget, "", state_file=state_file, payload={"issue_number": 7}, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. The probe call
# ---------------------------------------------------------------------------


def test_probe_runs_select_against_an_empty_diff_in_the_repo_root(
    widget: Path, fake_exe: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(SELECTED_EMPTY))
    target = SelectionTarget(fake_exe, "owner/repo-a", "origin/main")
    assert selection_wrapper.probe_selection(target, widget) is None
    [(argv, kwargs)] = runner.calls
    assert argv == [
        str(fake_exe),
        "select",
        "--repo",
        "owner/repo-a",
        "--base",
        "origin/main",
        "--head",
        "origin/main",
        "--format",
        "json",
    ]
    assert kwargs == {"cwd": widget, "timeout_seconds": SELECT_PROBE_TIMEOUT_SECONDS}
    assert SELECT_PROBE_TIMEOUT_SECONDS == 60


def test_worker_selection_probes_once_with_the_resolved_target(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(SELECTED_EMPTY))
    _worker_selection(widget, _state_file(tmp_path))
    [(argv, kwargs)] = runner.calls
    assert argv[1:] == [
        "select",
        "--repo",
        "local/widget",
        "--base",
        "main",
        "--head",
        "main",
        "--format",
        "json",
    ]
    assert kwargs["cwd"] == widget


# ---------------------------------------------------------------------------
# 2. A full verdict falls back; 3. its positive control
# ---------------------------------------------------------------------------


def test_a_full_verdict_falls_back_to_the_hand_picked_command(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    assert _worker_selection(widget, state_file) is None
    [payload] = _events(state_file, UNAVAILABLE)
    assert payload["reason"] == SELECTION_FULL == "selection-full"
    assert "map-stale" in payload["detail"]
    assert "empty diff" in payload["detail"]
    assert payload["issue_number"] == 7

    values = prompt_test_command_values(RUNNER, widget, selection=None)
    assert "ci-fleet" not in values["targeted_test_command"]
    assert all(m in values["test_execution_contract"] for m in EXECUTION_CONTRACT_MARKERS)


def test_a_selected_verdict_keeps_the_wrapper(
    tmp_path: Path, widget: Path, fake_exe: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    """Positive control for the fallback: the same path, a selector that can narrow."""
    answering(_ok(SELECTED_EMPTY))
    state_file = _state_file(tmp_path)
    target = _worker_selection(widget, state_file)
    assert target == SelectionTarget(fake_exe, "local/widget", "main")
    assert _events(state_file, UNAVAILABLE) == []

    command = prompt_test_command_values(RUNNER, widget, selection=target)["targeted_test_command"]
    assert shlex.split(command)[0] == fake_exe.as_posix()
    assert " test --repo " in command
    assert "--context worker" in command


def test_a_full_verdict_without_a_reason_still_falls_back(
    widget: Path, fake_exe: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    answering(_ok(json.dumps({"mode": "full"})))
    unavailable = selection_wrapper.probe_selection(
        SelectionTarget(fake_exe, "local/widget", "main"), widget
    )
    assert isinstance(unavailable, SelectionUnavailable)
    assert unavailable.reason == SELECTION_FULL
    assert "none given" in unavailable.detail


def test_the_real_full_payload_is_recognised(
    widget: Path, fake_exe: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    """``tests/fixtures/ci_fleet_select_full.json`` is ``ci-fleet select --format json``'s output.

    Captured live against this repository (empty diff, stale selection map); the
    unit tests above use the two keys the probe reads, this one pins them to the
    selector's real shape.
    """
    payload = (Path(__file__).parent / "fixtures" / "ci_fleet_select_full.json").read_text(
        encoding="utf-8"
    )
    reason = json.loads(payload)["reason"]
    answering(_ok(payload))
    unavailable = selection_wrapper.probe_selection(
        SelectionTarget(fake_exe, "local/widget", "main"), widget
    )
    assert isinstance(unavailable, SelectionUnavailable)
    assert unavailable.reason == SELECTION_FULL
    assert reason in unavailable.detail


# ---------------------------------------------------------------------------
# 4. Everything else fails open
# ---------------------------------------------------------------------------

_FAIL_OPEN = [
    pytest.param(
        RunResult(returncode=1, stdout=FULL_MAP_STALE, stderr="boom", error="command exited 1"),
        id="non-zero-exit",
    ),
    pytest.param(
        RunResult(returncode=None, stdout=FULL_MAP_STALE, stderr="", timed_out=True),
        id="timed-out",
    ),
    pytest.param(
        RunResult(returncode=None, stdout="", stderr="", error="not found"), id="missing-binary"
    ),
    pytest.param(_ok(""), id="empty-stdout"),
    pytest.param(_ok("map is stale, running full"), id="not-json"),
    pytest.param(_ok(json.dumps([{"mode": "full", "reason": "map-stale"}])), id="json-array"),
    pytest.param(_ok(json.dumps("full")), id="json-string"),
    pytest.param(_ok(json.dumps({"reason": "map-stale"})), id="no-mode"),
]


@pytest.mark.parametrize("result", _FAIL_OPEN)
def test_an_unreadable_answer_keeps_the_wrapper(
    tmp_path: Path,
    widget: Path,
    fake_exe: Path,
    answering: Callable[[RunResult], ExeCalls],
    result: RunResult,
) -> None:
    """Each answer here would read as ``full`` if the guard on it were missing."""
    answering(result)
    state_file = _state_file(tmp_path)
    assert _worker_selection(widget, state_file) == SelectionTarget(
        fake_exe, "local/widget", "main"
    )
    assert _events(state_file, UNAVAILABLE) == []


# ---------------------------------------------------------------------------
# 5. The kill switch
# ---------------------------------------------------------------------------


def test_probe_false_never_runs_select(
    tmp_path: Path, widget: Path, fake_exe: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    target = _worker_selection(widget, state_file, probe=False)
    assert target == SelectionTarget(fake_exe, "local/widget", "main")
    assert runner.calls == []
    assert _events(state_file, UNAVAILABLE) == []


# ---------------------------------------------------------------------------
# 6. Config, and both prompt writers
# ---------------------------------------------------------------------------


def test_the_probe_is_on_by_default() -> None:
    assert DispatchConfig().worker_selection_probe is True
    assert build_config_from_data({}).dispatch.worker_selection_probe is True


def test_a_yaml_null_builds_the_default() -> None:
    raw = yaml.safe_load("dispatch:\n  worker_selection_probe:\n")
    assert raw == {"dispatch": {"worker_selection_probe": None}}
    assert build_config_from_data(raw).dispatch.worker_selection_probe is True


def test_false_is_honoured() -> None:
    raw = yaml.safe_load("dispatch:\n  worker_selection_probe: false\n")
    assert build_config_from_data(raw).dispatch.worker_selection_probe is False


def _app(tmp_path: Path, *, probe: bool) -> OrchestratorApp:
    repo = _repo(tmp_path / "widget")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "w"\nversion = "0"\n'
        '[project.optional-dependencies]\ndev = ["pytest"]\n',
        encoding="utf-8",
    )
    config = build_config_from_data({"dispatch": {"worker_selection_probe": probe}})
    return OrchestratorApp(repo, runtime_paths(repo, config.runtime.state_dir), config, gh=None)


def _render_worker_prompt(app: OrchestratorApp) -> str:
    return app._write_worker_prompt(ISSUE).read_text(encoding="utf-8")


def _render_rework(app: OrchestratorApp) -> str:
    return _render_rework_prompt(
        app.paths.state_file, PR, 1, "A note.", app.config, repo_root=app.repo_root
    )


_WRITERS = pytest.mark.parametrize(
    "render", [_render_worker_prompt, _render_rework], ids=["worker-prompt", "rework-prompt"]
)


@_WRITERS
def test_a_writer_with_the_probe_off_never_runs_select(
    tmp_path: Path,
    fake_exe: Path,
    answering: Callable[[RunResult], ExeCalls],
    render: Callable[[OrchestratorApp], str],
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    text = render(_app(tmp_path, probe=False))
    target = SelectionTarget(fake_exe, "local/widget", "main")
    assert selection_wrapper.worker_command(target, RUNNER, "-q --tb=short") in text
    assert runner.calls == []


@_WRITERS
def test_a_writer_with_the_probe_on_falls_back_on_a_full_verdict(
    tmp_path: Path,
    answering: Callable[[RunResult], ExeCalls],
    render: Callable[[OrchestratorApp], str],
) -> None:
    """Control for the test above: the same writer, probe on, the same full answer."""
    runner = answering(_ok(FULL_MAP_STALE))
    app = _app(tmp_path, probe=True)
    text = render(app)
    assert "--context worker" not in text
    assert f"{RUNNER} <impacted test files> -q --tb=short" in text
    assert EXECUTION_CONTRACT_MARKERS[0] in text
    assert runner.subcommands() == ["select"]
    [payload] = _events(app.paths.state_file, UNAVAILABLE)
    assert payload["reason"] == SELECTION_FULL


# ---------------------------------------------------------------------------
# 7. The contract text no longer promises output the console never shows
# ---------------------------------------------------------------------------


def test_the_selected_contract_does_not_promise_console_output() -> None:
    assert "it prints" not in _selected_contract("x")
    assert "prints `full`" not in _selected_contract("x")


@pytest.mark.parametrize("template", ["worker.md", "worker_claude_code.md", "rework.md"])
def test_a_selected_prompt_still_passes_the_contract_guard(template: str) -> None:
    target = SelectionTarget(Path("/venv/bin/ci-fleet"), "owner/repo-a", "origin/main")
    values = prompt_test_command_values(RUNNER, None, selection=target)
    prompt = render_prompt(template, {**ISSUE_VALUES, **values})
    assert_execution_contract(prompt)
    assert all(m in prompt for m in SELECTION_CONTRACT_MARKERS)
    assert "it prints" not in prompt


# ---------------------------------------------------------------------------
# 8. The merge gate is never probed
# ---------------------------------------------------------------------------


def test_the_merge_gate_never_runs_select(
    widget: Path, fake_exe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ExeCalls(fake_exe, {"select": _ok(FULL_MAP_STALE)})
    monkeypatch.setattr(selection_wrapper, "run_captured", runner)
    selection = selection_wrapper.gate_selection(
        widget, ["pytest", "-q"], head="h" * 40, base="b" * 40
    )
    assert selection.mode == selection_wrapper.SHADOW
    assert runner.subcommands() == ["nightly"]


# ---------------------------------------------------------------------------
# 9. One probe per pass per repo, one row per change (issue #2921)
# ---------------------------------------------------------------------------
#
# ``worker_selection`` runs for every candidate on every pass, including candidates the
# concurrency cap defers. A *pass* is the ambient correlation id (``loop()`` wraps each
# pass in ``correlation_context()``), so each ``with correlation_context():`` block below
# is one pass.

NO_MAP = json.dumps({"mode": "full", "reason": "no-map"})
DISABLED_KIND = "worker_test_selection_disabled"


def _rows(state_file: Path, kind: str) -> list[tuple[str, dict]]:
    """``(level, payload)`` of every ``kind`` row, oldest first."""
    db = state_file.parent / "events.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT level, payload FROM events WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()
    finally:
        conn.close()
    return [(level, json.loads(payload)) for level, payload in rows]


@pytest.fixture(autouse=True)
def _empty_pass_memo() -> None:
    """Tests must not share the module-level once-per-pass memo."""
    from charlie_work import selection_answer

    selection_answer._memo.clear()


@pytest.mark.parametrize(
    ("answer", "rows"), [(FULL_MAP_STALE, 1), (SELECTED_EMPTY, 0)], ids=["full", "selected"]
)
def test_n_prompts_in_one_pass_run_select_once(
    tmp_path: Path,
    widget: Path,
    answering: Callable[[RunResult], ExeCalls],
    answer: str,
    rows: int,
) -> None:
    runner = answering(_ok(answer))
    state_file = _state_file(tmp_path)
    with correlation_context():
        results = [_worker_selection(widget, state_file) for _ in range(6)]
    assert runner.subcommands() == ["select"]
    assert len(set(results)) == 1, "every prompt of the pass gets the same answer"
    assert len(_rows(state_file, UNAVAILABLE)) == rows


def test_every_prompt_writer_shares_the_one_probe_of_a_pass(
    tmp_path: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    """The worker and rework writers both reach ``worker_selection``; one pass, one probe."""
    runner = answering(_ok(FULL_MAP_STALE))
    app = _app(tmp_path, probe=True)
    with correlation_context():
        app._write_worker_prompt(ISSUE)
        app._write_worker_prompt({**ISSUE, "number": 2})
        text = _render_rework(app)
        app._write_worker_prompt({**ISSUE, "number": 3})
    assert f"{RUNNER} <impacted test files> -q --tb=short" in text
    assert runner.subcommands() == ["select"]
    assert len(_rows(app.paths.state_file, UNAVAILABLE)) == 1


def test_a_new_pass_probes_again_but_an_unchanged_full_writes_no_new_row(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    for _ in range(3):
        with correlation_context():
            for _ in range(4):
                _worker_selection(widget, state_file)
    assert runner.subcommands() == ["select"] * 3, "one probe per pass"
    [(level, payload)] = _rows(state_file, UNAVAILABLE)
    assert level == "warning", "the first selection-full stays a warning"
    assert payload["reason"] == SELECTION_FULL
    assert "map-stale" in payload["detail"]


def test_a_changed_reason_writes_exactly_one_new_row(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    for answer in (FULL_MAP_STALE, FULL_MAP_STALE, NO_MAP, NO_MAP, NO_MAP):
        runner.answers["select"] = _ok(answer)
        with correlation_context():
            _worker_selection(widget, state_file)
            _worker_selection(widget, state_file)
    rows = _rows(state_file, UNAVAILABLE)
    assert [("map-stale" in p["detail"], "no-map" in p["detail"]) for _, p in rows] == [
        (True, False),
        (False, True),
    ]
    assert {level for level, _ in rows} == {"warning"}


def test_a_full_answer_that_recurs_after_a_usable_one_is_a_fresh_row(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    for answer in (FULL_MAP_STALE, SELECTED_EMPTY, SELECTED_EMPTY, FULL_MAP_STALE):
        runner.answers["select"] = _ok(answer)
        with correlation_context():
            _worker_selection(widget, state_file)
    assert len(_rows(state_file, UNAVAILABLE)) == 2


def test_the_probe_kill_switch_is_not_answered_from_a_probed_memo(
    tmp_path: Path,
    widget: Path,
    fake_exe: Path,
    answering: Callable[[RunResult], ExeCalls],
) -> None:
    """``probe=False`` is part of the memo key: it still turns the probe off entirely."""
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    wrapper = SelectionTarget(fake_exe, "local/widget", "main")
    with correlation_context():
        assert _worker_selection(widget, state_file, probe=True) is None
        assert _worker_selection(widget, state_file, probe=False) == wrapper
        assert _worker_selection(widget, state_file, probe=True) is None
    assert runner.subcommands() == ["select"]

    runner.calls.clear()
    with correlation_context():
        assert _worker_selection(widget, state_file, probe=False) == wrapper
        assert runner.calls == []
        assert _worker_selection(widget, state_file, probe=True) is None
    assert runner.subcommands() == ["select"]


def test_each_repo_in_a_pass_probes_once(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    runner = answering(_ok(FULL_MAP_STALE))
    gadget = _repo(tmp_path / "gadget")
    with correlation_context():
        for repo in (widget, gadget, widget, gadget):
            _worker_selection(repo, _state_file(tmp_path / repo.name))
    assert runner.subcommands() == ["select", "select"]
    assert [argv[3] for argv, _ in runner.calls] == ["local/widget", "local/gadget"]
    for repo in (widget, gadget):
        assert len(_rows(_state_file(tmp_path / repo.name), UNAVAILABLE)) == 1


def test_a_disabled_repo_reports_once_across_passes(tmp_path: Path, fake_exe: Path) -> None:
    repo = _repo(tmp_path / "widget")
    (repo / "ci-fleet.toml").write_text("[test_selection]\nenabled = false\n", encoding="utf-8")
    state_file = _state_file(tmp_path)
    for _ in range(3):
        with correlation_context():
            for _ in range(2):
                assert _worker_selection(repo, state_file) is None
    assert [level for level, _ in _rows(state_file, DISABLED_KIND)] == ["info"]
    assert _rows(state_file, UNAVAILABLE) == []


def test_outside_a_pass_every_call_resolves_but_the_row_is_still_written_once(
    tmp_path: Path, widget: Path, answering: Callable[[RunResult], ExeCalls]
) -> None:
    """No correlation id, no memo (today's behaviour); the change baseline still holds."""
    runner = answering(_ok(FULL_MAP_STALE))
    state_file = _state_file(tmp_path)
    for _ in range(3):
        _worker_selection(widget, state_file)
    assert runner.subcommands() == ["select"] * 3
    assert len(_rows(state_file, UNAVAILABLE)) == 1
