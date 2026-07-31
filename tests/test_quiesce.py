"""Tests for the fleet quiescence predicate (src/charlie_work/quiesce.py).

Covers the three real defects the PowerShell-prose version of this gate had
(self-match force-killing the operator's shell, name-based matching hitting
the Windsurf/Devin.exe IDE, and a multi-process lineage being under-reported)
plus baseline robustness: empty input, cyclic parent chains, non-matching
patterns, and malformed regexes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from charlie_work.quiesce import (
    ProcessInfo,
    QuiesceReport,
    assert_quiescent,
    check_quiescence,
    list_processes,
    self_process_chain,
)

# A single pattern standing in for the config-supplied fleet-process regexes
# a real caller would pass (never hardcoded inside quiesce.py itself).
_SUPERVISE_PATTERN = r"fleet supervise"
_REVIEWER_PATTERN = r"devin --model \S+ -p --prompt-file"


def _proc(pid: int, ppid: int, name: str, command_line: str) -> ProcessInfo:
    return ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line)


# ---------------------------------------------------------------------------
# Defect #1: the observer is in the observed set.
# ---------------------------------------------------------------------------


def test_self_pid_excluded_even_when_its_own_command_line_matches_pattern() -> None:
    """A kill-loop's own shell command line can literally contain the pattern
    text it is searching for (real incident: this force-killed the operator's
    shell). The self PID must never appear in ``matched``.
    """
    processes = [
        _proc(
            100, 1, "powershell.exe", "powershell -Command Get-CimInstance ... 'fleet supervise'"
        ),
    ]

    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=processes, self_pid=100)

    assert report.ok is True
    assert report.matched == ()
    assert 100 in report.excluded_pids
    assert "quiescent" in report.summary


def test_ancestor_two_levels_up_is_excluded() -> None:
    """A parent shell's command line can also contain the pattern (e.g. a
    wrapper script that echoes the command it's about to run). Walking the
    full ancestor chain, not just the immediate parent, must exclude it too.
    """
    # self (300) -> parent (200) -> grandparent (100), grandparent's command
    # line contains the search pattern.
    processes = [
        _proc(100, 1, "cmd.exe", "cmd /c run-quiesce-check.bat -- fleet supervise"),
        _proc(200, 100, "powershell.exe", "powershell -File check.ps1"),
        _proc(300, 200, "python.exe", "python quiesce_cli.py"),
    ]

    chain = self_process_chain(300, processes)
    # pid 1 (the grandparent's stated ppid) has no entry of its own in
    # ``processes`` -- it is discovered as the final ancestor and included
    # too. That's fine: it can never be *matched* either, since matching
    # only ever iterates ``processes``, and pid 1 has no row there.
    assert chain == frozenset({300, 200, 100, 1})

    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=processes, self_pid=300)

    assert report.ok is True
    assert report.matched == ()
    assert report.excluded_pids == frozenset({100, 200, 300, 1})


# ---------------------------------------------------------------------------
# Defect #2: name-based matching hits the Windsurf/Devin.exe IDE.
# ---------------------------------------------------------------------------


def test_ide_devin_acp_process_is_not_reported() -> None:
    """Windsurf's ``devin.exe acp --agent-type summarizer`` language server has
    ``name == "devin.exe"`` but is not a fleet-reviewer invocation. Matching
    must be command-line-only, so this must never be reported regardless of
    process name.
    """
    processes = [
        _proc(500, 1, "devin.exe", "devin.exe acp --agent-type summarizer"),
        _proc(501, 1, "devin.exe", "devin.exe acp"),
    ]

    report = assert_quiescent(patterns=[_REVIEWER_PATTERN], processes=processes, self_pid=9999)

    assert report.ok is True
    assert report.matched == ()


def test_fleet_reviewer_invocation_is_matched_by_command_line() -> None:
    """The actual fleet reviewer invocation must be caught: a distinct command
    line, not the editor's name, is what identifies it.
    """
    reviewer = _proc(
        502,
        1,
        "devin.exe",
        r"devin --model glm-5.2 -p --prompt-file C:\repo\.var\cross-family-prompt.md",
    )
    processes = [
        _proc(500, 1, "devin.exe", "devin.exe acp --agent-type summarizer"),
        reviewer,
    ]

    report = assert_quiescent(patterns=[_REVIEWER_PATTERN], processes=processes, self_pid=9999)

    assert report.ok is False
    assert report.matched == (reviewer,)
    assert "502" in report.summary


def test_name_matching_pattern_is_not_reported_when_command_line_does_not() -> None:
    """Matching is command-line-only. A process whose *name* contains the
    pattern text, but whose command line does not, must never be reported.

    Without this test, defect #2 is only pinned by accident: both
    `_REVIEWER_PATTERN` and `_SUPERVISE_PATTERN` happen to never match any
    ``name`` value used elsewhere in this file, so a regression that widened
    matching to `rx.search(proc.command_line) or rx.search(proc.name)` would
    pass every other test here. This process's name is deliberately built
    from the supervise pattern text while its command line is unrelated,
    so that mutation would flip this assertion from ``ok=True`` to
    ``ok=False``.
    """
    impostor = _proc(600, 1, "fleet supervise.exe", r"C:\tools\unrelated.exe --idle")

    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=[impostor], self_pid=9999)

    assert report.ok is True
    assert report.matched == ()
    assert report.excluded_pids == frozenset({9999})


# ---------------------------------------------------------------------------
# Defect #3: one logical process is several OS processes.
# ---------------------------------------------------------------------------


def test_full_process_lineage_is_reported_not_just_one_pid() -> None:
    """``fleet supervise`` is a lineage: charlie.exe -> python.exe -> python.exe,
    under a wscript.exe trampoline. All three matching lineage members must
    be reported; the non-matching trampoline must not be.
    """
    trampoline = _proc(10, 1, "wscript.exe", r"wscript.exe C:\fleet\launch.vbs")
    launcher = _proc(20, 10, "charlie.exe", "charlie.exe fleet supervise --repo charlie-work")
    outer_py = _proc(
        30, 20, "python.exe", "python -m charlie_work fleet supervise --repo charlie-work"
    )
    inner_py = _proc(40, 30, "python.exe", "python -c fleet supervise worker loop")
    processes = [trampoline, launcher, outer_py, inner_py]

    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=processes, self_pid=9999)

    assert report.ok is False
    assert set(report.matched) == {launcher, outer_py, inner_py}
    assert trampoline not in report.matched


# ---------------------------------------------------------------------------
# Baseline robustness.
# ---------------------------------------------------------------------------


def test_empty_process_list_is_quiescent() -> None:
    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=[], self_pid=1234)

    assert report.ok is True
    assert report.matched == ()
    # self_pid is still recorded as excluded even though it has no entry in
    # the (empty) snapshot -- the chain always contains at least itself.
    assert report.excluded_pids == frozenset({1234})


def test_cyclic_parent_chain_terminates() -> None:
    """A malformed/adversarial snapshot where PPIDs form a cycle must not hang
    the walk. self_process_chain must return the cycle's members and stop.
    """
    processes = [
        _proc(1, 2, "a.exe", "a"),
        _proc(2, 3, "b.exe", "b"),
        _proc(3, 1, "c.exe", "c"),  # closes the cycle back to pid 1
    ]

    chain = self_process_chain(1, processes)

    assert chain == frozenset({1, 2, 3})


def test_self_referential_parent_terminates() -> None:
    """A process that is its own parent (ppid == pid) is a degenerate cycle
    of length one and must terminate immediately.
    """
    processes = [_proc(7, 7, "weird.exe", "weird")]

    chain = self_process_chain(7, processes)

    assert chain == frozenset({7})


def test_pattern_matching_nothing_is_quiescent() -> None:
    processes = [
        _proc(1, 0, "explorer.exe", "explorer.exe"),
        _proc(2, 1, "chrome.exe", "chrome.exe --type=renderer"),
    ]

    report = assert_quiescent(
        patterns=[r"nothing-should-match-this-xyz"], processes=processes, self_pid=999
    )

    assert report.ok is True
    assert report.matched == ()


def test_pattern_with_regex_metacharacters_matches_literally_and_does_not_raise() -> None:
    """A pattern containing valid-but-special regex characters (``.``, ``-``,
    parens) must still work as a regex, not raise, and match correctly.
    """
    reviewer = _proc(
        42,
        1,
        "devin.exe",
        r"devin --model glm-5.2 -p --prompt-file C:\repo\prompt (cross-family).md",
    )
    processes = [reviewer]

    report = assert_quiescent(
        patterns=[r"glm-5.2 -p --prompt-file"], processes=processes, self_pid=999
    )

    assert report.ok is False
    assert report.matched == (reviewer,)
    assert report.invalid_patterns == ()


def test_invalid_regex_pattern_is_skipped_not_raised() -> None:
    """A caller-supplied pattern that is not valid regex (e.g. a Windows path
    with a bad escape like ``\\U``) must not crash the check. It is skipped
    and reported back via ``invalid_patterns``, while other valid patterns in
    the same call still apply normally.
    """
    reviewer = _proc(42, 1, "devin.exe", "devin --model glm-5.2 -p --prompt-file C:\\x.md")
    processes = [reviewer]

    # r"\Users" is an invalid regex escape (\U is not a recognized escape).
    report = assert_quiescent(
        patterns=[r"\Users", _REVIEWER_PATTERN], processes=processes, self_pid=999
    )

    assert report.ok is False
    assert report.matched == (reviewer,)
    assert report.invalid_patterns == (r"\Users",)
    assert "invalid pattern" in report.summary


def test_qu_report_is_frozen_dataclass() -> None:
    """Config/value-object types in this repo are frozen dataclasses
    (CLAUDE.md invariant); QuiesceReport must not be mutable.
    """
    report = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="x")

    with pytest.raises(AttributeError):
        report.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# list_processes() -- default Windows lister.
# ---------------------------------------------------------------------------


def test_list_processes_non_windows_returns_empty_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    processes, error = list_processes()

    assert processes == ()
    assert error is not None
    assert "win32" in error


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_list_processes_windows_powershell_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    processes, error = list_processes()

    assert processes == ()
    assert error is not None
    assert "powershell" in error


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_list_processes_windows_parses_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = [
        {
            "ProcessId": 111,
            "ParentProcessId": 1,
            "Name": "charlie.exe",
            "CommandLine": "charlie.exe fleet supervise",
        },
        {
            "ProcessId": 222,
            "ParentProcessId": 111,
            "Name": "python.exe",
            "CommandLine": "python -m charlie_work fleet supervise",
        },
    ]

    class _FakeResult:
        returncode = 0
        stdout = json.dumps(sample)
        stderr = ""

    monkeypatch.setattr("shutil.which", lambda _name: "C:\\Windows\\powershell.exe")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeResult())

    processes, error = list_processes()

    assert error is None
    assert processes == (
        ProcessInfo(
            pid=111, ppid=1, name="charlie.exe", command_line="charlie.exe fleet supervise"
        ),
        ProcessInfo(
            pid=222,
            ppid=111,
            name="python.exe",
            command_line="python -m charlie_work fleet supervise",
        ),
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_list_processes_windows_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired("powershell", 15)

    monkeypatch.setattr("shutil.which", lambda _name: "C:\\Windows\\powershell.exe")
    monkeypatch.setattr("subprocess.run", _raise)

    processes, error = list_processes()

    assert processes == ()
    assert error is not None
    assert "powershell invocation failed" in error


# ---------------------------------------------------------------------------
# check_quiescence() -- lister + assert_quiescent glued together.
# ---------------------------------------------------------------------------


def test_check_quiescence_fails_closed_on_lister_error() -> None:
    def _broken_lister() -> tuple[list[ProcessInfo], str | None]:
        return [], "simulated CIM query failure"

    report = check_quiescence(patterns=[_SUPERVISE_PATTERN], self_pid=1, lister=_broken_lister)

    assert report.ok is False
    assert "simulated CIM query failure" in report.summary


def test_check_quiescence_delegates_to_assert_quiescent_on_success() -> None:
    reviewer = _proc(50, 1, "devin.exe", "devin --model glm-5.2 -p --prompt-file x.md")

    def _fake_lister() -> tuple[list[ProcessInfo], str | None]:
        return [reviewer], None

    report = check_quiescence(patterns=[_REVIEWER_PATTERN], self_pid=999, lister=_fake_lister)

    assert report.ok is False
    assert report.matched == (reviewer,)


@pytest.mark.skipif(sys.platform != "win32", reason="list_processes is win32-only")
def test_list_processes_actually_runs_against_this_host() -> None:
    """Execute the real PowerShell snapshot -- no injected lister.

    Every other test in this file substitutes a fake ``lister``, which is why a
    PowerShell incompatibility survived a fully green suite: ``ConvertTo-Json
    -AsArray`` is PowerShell 6+, so on Windows PowerShell 5.1 the command failed
    with a ParameterBindingException and ``check_quiescence`` fail-closed on
    *every* invocation. A gate that can only ever answer "not quiescent" is
    indistinguishable from a working gate right up until you need it to say yes.

    This asserts the level the fake-lister tests structurally cannot: that the
    real command runs on the real interpreter and returns parseable data. The
    running interpreter must appear in its own snapshot, which also pins the
    ProcessId/CommandLine field names the parser depends on.
    """
    processes, error = list_processes()

    assert error is None, f"real process listing failed: {error}"
    assert processes, "process listing returned no processes"

    own = [p for p in processes if p.pid == os.getpid()]
    assert own, "the running interpreter did not appear in its own process snapshot"
    assert own[0].command_line, "own process reported an empty command line"


@pytest.mark.skipif(sys.platform != "win32", reason="list_processes is win32-only")
def test_check_quiescence_can_report_quiescent_against_the_real_lister() -> None:
    """A pattern matching nothing must yield ok=True through the real lister.

    The complement of the test above. Together they pin both answers: the
    -AsArray bug made ok=True unreachable, so a test asserting only the
    not-quiescent path would have passed against the broken implementation.
    """
    report = check_quiescence(patterns=[r"pattern-that-matches-no-command-line-\d{9}"])

    assert report.ok is True, f"expected quiescent, got: {report.summary}"
    assert report.matched == ()
