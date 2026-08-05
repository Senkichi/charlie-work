"""Regression tests for ``GitHub.pr_create``.

`pr_create` shipped passing ``--json number`` to ``gh pr create``, which has no
such flag. ``gh`` exited non-zero at argument parsing, before contacting the
API, so the method could never succeed and no PR was ever created -- the
orchestrator's whole "adopt a branch a worker pushed but could not open a PR
for" recovery lane (#935) was inert for every input.

Nothing caught it because every test in the suite substitutes a *fake* `gh`
object whose `pr_create` is a Python method returning a canned number. A fake
built alongside the caller is correct by construction and cannot disagree with
the real CLI about what flags exist -- no amount of coverage through that fake
would have found this. So the load-bearing test here is
``test_every_flag_pr_create_sends_is_accepted_by_the_installed_gh``, which
checks the arguments against ``gh`` itself rather than against our model of it.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import types

import pytest

from charlie_work.github import GitHub, _pr_number_from_url

_URL = "https://github.com/Senkichi/charlie-work/pull/1234"


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capture(monkeypatch: pytest.MonkeyPatch, **result: object) -> list[list[str]]:
    """Patch subprocess.run inside github.py and record every argv it sees."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> _FakeCompleted:
        calls.append(list(command))
        return _FakeCompleted(**result)  # type: ignore[arg-type]

    monkeypatch.setattr("charlie_work.github.subprocess.run", fake_run)
    return calls


def _client(tmp_path: pathlib.Path) -> GitHub:
    return GitHub(repo_root=tmp_path)


# --------------------------------------------------------------------------
# The bug itself
# --------------------------------------------------------------------------


def test_pr_create_does_not_send_a_json_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gh pr create` is a mutation and reports its result by printing a URL;
    it has no --json. Sending one makes gh fail at argument parsing."""
    calls = _capture(monkeypatch, returncode=0, stdout=_URL)
    _client(tmp_path).pr_create(head="agent/issue-1", base="main", title="t", body="b")

    assert len(calls) == 1
    argv = calls[0]
    assert "--json" not in argv
    # Control: the command really is the one under test, so the absence above is
    # about pr_create's arguments and not about an empty/short argv.
    assert argv[:3] == ["gh", "pr", "create"]
    assert "--head" in argv and "--base" in argv


def test_every_flag_pr_create_sends_is_accepted_by_the_installed_gh(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test that would have caught this.

    Our fakes agree with us by construction; only `gh` can contradict us. This
    asserts every long flag pr_create sends appears in `gh pr create --help`.
    """
    if shutil.which("gh") is None:
        pytest.skip("gh CLI not installed")

    # Read the help text BEFORE patching. `_capture` replaces
    # `charlie_work.github.subprocess.run`, and that attribute *is* the shared
    # subprocess module's `run`, so a help call made afterwards would be served
    # by the fake and this test would silently validate our own stub against
    # itself. (Caught by the control below on the first run -- it asserted a
    # real help text and got the fake's PR URL.)
    help_text = subprocess.run(
        ["gh", "pr", "create", "--help"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    # Control: if the help text is empty or unrecognisable, every membership
    # test below would be vacuous. Assert we actually got help output for the
    # right subcommand before drawing any conclusion from it.
    assert "--head" in help_text, "unexpected `gh pr create --help` output; test is blind"

    calls = _capture(monkeypatch, returncode=0, stdout=_URL)
    _client(tmp_path).pr_create(head="agent/issue-1", base="main", title="t", body="b")
    sent = {tok for tok in calls[0] if tok.startswith("--")}

    unsupported = sorted(flag for flag in sent if flag not in help_text)
    assert not unsupported, (
        f"pr_create sends flags the installed gh does not accept: {unsupported}. "
        "gh exits non-zero at argument parsing before contacting the API, so "
        "pr_create would silently never create a PR."
    )


# --------------------------------------------------------------------------
# Parsing the number back out
# --------------------------------------------------------------------------


def test_pr_create_returns_the_number_from_the_url(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture(monkeypatch, returncode=0, stdout=_URL + "\n")
    assert _client(tmp_path).pr_create(head="h", base="main", title="t", body="b") == 1234


def test_pr_create_prefers_the_last_url_in_the_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gh prints progress chatter before the URL, and a title or body echoed
    into it can legitimately contain another PR link ("supersedes .../pull/900").
    The URL gh appends last is the one it created."""
    noisy = (
        "Creating pull request for agent/issue-1 into main in Senkichi/charlie-work\n"
        "note: supersedes https://github.com/Senkichi/charlie-work/pull/900\n"
        f"{_URL}\n"
    )
    _capture(monkeypatch, returncode=0, stdout=noisy)
    assert _client(tmp_path).pr_create(head="h", base="main", title="t", body="b") == 1234


@pytest.mark.parametrize(
    "output, expected",
    [
        (_URL, 1234),
        (f"{_URL}\n", 1234),
        ("https://github.com/o/r/pull/7", 7),
        ("", None),
        ("no url here", None),
        ("https://github.com/o/r/issues/12", None),
    ],
)
def test_pr_number_from_url(output: str, expected: int | None) -> None:
    assert _pr_number_from_url(output) == expected


# --------------------------------------------------------------------------
# Failure paths return values, never raise
# --------------------------------------------------------------------------


def test_pr_create_returns_none_when_gh_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture(monkeypatch, returncode=1, stdout="", stderr="pull request already exists")
    assert _client(tmp_path).pr_create(head="h", base="main", title="t", body="b") is None


def test_pr_create_logs_the_stderr_when_gh_fails(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The caller only sees None. Without the reason in the log, "gh is not
    authenticated", "GitHub rejected it", and "we sent a bad flag" are
    indistinguishable -- which is what made the original bug expensive."""
    _capture(monkeypatch, returncode=1, stdout="", stderr="unknown flag: --json")
    with caplog.at_level("WARNING"):
        _client(tmp_path).pr_create(head="h", base="main", title="t", body="b")
    assert "unknown flag: --json" in caplog.text


def test_pr_create_returns_none_when_output_has_no_url(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 with no URL must not be reported as a created PR. Returning a
    bogus number would put an unusable pr_number into state.json."""
    _capture(monkeypatch, returncode=0, stdout="something unexpected")
    assert _client(tmp_path).pr_create(head="h", base="main", title="t", body="b") is None


def test_pr_create_is_a_no_op_under_dry_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, returncode=0, stdout=_URL)
    assert (
        GitHub(repo_root=tmp_path, dry_run=True).pr_create(
            head="h", base="main", title="t", body="b"
        )
        == 0
    )
    assert calls == []


def test_pr_create_returns_none_when_gh_is_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`allow_failure=True` must absorb a missing binary as a value, per the
    repo invariant that external-process errors come back as values."""

    def boom(command: list[str], **kwargs: object) -> types.SimpleNamespace:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr("charlie_work.github.subprocess.run", boom)
    assert _client(tmp_path).pr_create(head="h", base="main", title="t", body="b") is None
