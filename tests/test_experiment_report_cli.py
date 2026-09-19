"""CLI-layer tests for ``charlie experiment-report`` (issue #1701).

Split out of ``test_experiment_report.py`` under the repo's 800-line
file-size cap (issue #1442).  Covers the command layer's contract: the
subparser's own help text, the read-only gates (missing events.db,
unmigrated events.jsonl, cold-open byte-identical state dir), the window
flags plumbage, argument error paths, ``--json`` output, and the
state.json merge-coverage handoff into ``build_report``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _cli_fixtures import _FakeGitHub, _make_repo
from _experiment_report_fixtures import (
    KEY,
    _repo_with_db,
    _state_dir_snapshot,
    _state_path,
)
from charlie_work import cli, instrumentation


def test_help_describes_read_only() -> None:
    """The experiment-report subparser's OWN help and description must
    state the read-only contract -- a top-level ``--help`` assertion is
    satisfied by unrelated commands' help text and cannot fail on the
    regression it claims to guard."""
    import argparse

    parser = cli.build_parser()
    subs = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    sub = subs.choices["experiment-report"]
    assert "read-only" in (sub.description or "").lower()
    pseudo = next(a for a in subs._choices_actions if a.dest == "experiment-report")
    assert "read-only" in (pseudo.help or "").lower()


def test_missing_events_db_fails_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only means: a missing events.db must not be created by the
    report (instrumentation._get_db would create+schema it on open)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    before = state_path.read_bytes()

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 1
    assert state_path.read_bytes() == before
    assert not (state_path.parent / "events.db").exists()


def test_report_performs_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """events.db, state.json, and every sibling file are byte-identical
    before and after a run on the COLD-OPEN path. Two subtleties make the
    comparison honest: (1) close_db BEFORE snapshotting so _get_db really
    re-opens the database inside the command -- a warm-cache run could
    mask a write on reopen; (2) close_db AFTER the run so any write that
    landed in the events.db-wal sidecar (WAL mode: a row write does not
    touch the main db file until a checkpoint) is folded back in before
    the byte comparison."""
    repo, state_path = _repo_with_db(tmp_path, monkeypatch)
    instrumentation.close_db(state_path)
    before = _state_dir_snapshot(state_path)

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 0
    out = capsys.readouterr().out
    assert 'arm "deep"' in out or '"deep"' in out
    assert "stopping rule" in out
    instrumentation.close_db(state_path)
    assert _state_dir_snapshot(state_path) == before


def test_command_reads_state_prs_for_merge_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: a PR that state.json marks merged but that has no merge
    event (the Aviator case) must count in merge_rate when run through the
    CLI -- and the run stays read-only (state dir byte-identical)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    instrumentation.log_event(
        state_path,
        "record_review",
        {
            "pr_number": 7,
            "issue_number": 1,
            "decision": "approved",
            "session_metrics": {KEY: "deep"},
        },
    )
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["prs"]["7"] = {
        "status": "merged",
        "merged_at": "2026-08-04T00:00:00Z",
        "issue_number": 1,
    }
    state_path.write_text(json.dumps(data), encoding="utf-8")
    instrumentation.close_db(state_path)
    before = _state_dir_snapshot(state_path)

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "merge events observed for 0 of 1 state-merged PRs" in out
    instrumentation.close_db(state_path)
    assert _state_dir_snapshot(state_path) == before


def test_unmigrated_events_jsonl_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unmigrated events.jsonl beside events.db must be refused (rc=1):
    reading through query_events would auto-migrate it -- a write. Both
    files must come out byte-identical."""
    repo, state_path = _repo_with_db(tmp_path, monkeypatch)
    instrumentation.close_db(state_path)
    jsonl = state_path.parent / "events.jsonl"
    jsonl.write_text(
        '{"ts": "2026-08-01T00:00:00Z", "kind": "record_review", "payload": {}}\n',
        encoding="utf-8",
    )
    db_bytes = (state_path.parent / "events.db").read_bytes()
    jsonl_bytes = jsonl.read_bytes()

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 1
    assert "refusing" in capsys.readouterr().out
    assert (state_path.parent / "events.db").read_bytes() == db_bytes
    assert jsonl.read_bytes() == jsonl_bytes


def test_exclude_window_flag_reaches_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--exclude-window START END plumbs through argparse into the report:
    an exclusion covering now drops every event (NO DATA), one covering
    only ancient history changes nothing."""
    repo, state_path = _repo_with_db(tmp_path, monkeypatch)
    instrumentation.close_db(state_path)

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "experiment-report",
            "--experiment",
            "review_effort",
            "--exclude-window",
            "2000-01-01T00:00:00Z",
            "2100-01-01T00:00:00Z",
        ]
    )
    assert rc == 0
    assert "NO DATA" in capsys.readouterr().out

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "experiment-report",
            "--experiment",
            "review_effort",
            "--exclude-window",
            "2000-01-01T00:00:00Z",
            "2000-01-02T00:00:00Z",
        ]
    )
    assert rc == 0
    assert 'arm "deep"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra_args", "needle"),
    [
        (["--since", "not-a-timestamp"], "not an ISO-8601 timestamp"),
        (["--until", "garbage"], "not an ISO-8601 timestamp"),
        (
            ["--since", "2026-08-10T00:00:00Z", "--until", "2026-08-01T00:00:00Z"],
            "is after --until",
        ),
        (
            [
                "--exclude-window",
                "2026-08-10T00:00:00Z",
                "2026-08-01T00:00:00Z",
            ],
            "is after end",
        ),
        (["--min-prs-per-arm", "0"], "must be a positive integer"),
        (["--min-prs-per-arm", "-3"], "must be a positive integer"),
    ],
)
def test_cli_window_and_min_prs_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    needle: str,
) -> None:
    """Every user-facing argument error is a CommandResult(False) -- rc=1
    with a named reason, never a traceback."""
    repo, state_path = _repo_with_db(tmp_path, monkeypatch)
    instrumentation.close_db(state_path)

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "experiment-report",
            "--experiment",
            "review_effort",
            *extra_args,
        ]
    )

    assert rc == 1
    assert needle in capsys.readouterr().out


def test_json_flag_emits_structured_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    instrumentation.log_event(
        state_path,
        "record_review",
        {
            "pr_number": 1,
            "issue_number": 1,
            "decision": "request_changes",
            "session_metrics": {KEY: "deep"},
        },
    )

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "experiment-report",
            "--experiment",
            "review_effort",
            "--json",
        ]
    )

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True
    data = parsed["data"]
    assert data["metrics_key"] == KEY
    assert data["arms"] == ["deep"]
    assert data["metrics"]["first_round_request_changes_rate"]["per_arm"]["deep"]["rate"] == 1.0
