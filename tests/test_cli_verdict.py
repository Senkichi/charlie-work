"""``charlie verdict`` (record_review) command tests.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _cli_fixtures import (
    _FakeGitHub,
    _make_repo,
)
from charlie_work import cli
from charlie_work.fleet_paths import fleet_dir
from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY
from charlie_work.workflow import ORCHESTRATOR_COMMENT_MARKER


def test_cli_verdict_missing_summary_file_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #357: a missing --summary-file must fail the CLI command (non-zero rc)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    missing_summary = repo / "missing-summary.md"

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(missing_summary),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "OS error" in captured.out
    assert "No such file or directory" in captured.out
    assert not (repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json").exists()


def test_cli_verdict_success_records_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: a readable summary file records the verdict and exits 0."""
    fake_gh = _FakeGitHub()
    monkeypatch.setattr(cli, "GitHub", lambda *a, **k: fake_gh)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    assert decision_path.exists()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "approved"
    assert decision["summary"] == "lgtm"

    # Issue #1268 (W11) AC4/AC5: post_verdict_comment defaults to True, so
    # this terminal decision must also post exactly one PR comment carrying
    # the round header -- see workflow.py's record_review comment gate and
    # tests/test_review_pr_comment.py for the dedicated coverage.
    assert len(fake_gh.pr_comment_calls) == 1
    posted_number, posted_body = fake_gh.pr_comment_calls[0]
    assert posted_number == 1
    assert posted_body.startswith(ORCHESTRATOR_COMMENT_MARKER + "\n")
    assert "## Fleet review - round 1 - approved" in posted_body


def test_cli_verdict_reviewed_head_flag_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #467: --reviewed-head is accepted and recorded with provenance."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
            "--reviewed-head",
            "sha-abc",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc"
    assert decision["reviewed_head_source"] == "live"
    assert "(head from live)" in captured.out


def test_cli_verdict_missing_summary_file_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --json, the missing-file failure is still a machine-parseable non-zero result."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    missing_summary = repo / "missing-summary.md"

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--json",
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(missing_summary),
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "OS error" in payload["message"]
    assert not (repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json").exists()


def test_cli_verdict_request_changes_derives_required_changes_from_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #792 AC-6, third producer path: `charlie verdict` has no
    --required-changes flag at all -- every request_changes/blocked verdict
    recorded through this CLI command arrives at record_review with
    required_changes=None, so real reviewer prose supplied via
    --summary-file must be derived into required_changes rather than
    persisted empty."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    prose = "The retry wrapper swallows the exception type; push a fix before merging."
    summary.write_text(prose, encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "request_changes",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "request_changes"
    assert decision["required_changes"] == [prose]
    assert decision["findings_channel"] == "derived"


def test_cli_verdict_request_changes_persists_vacuous_marker_for_legacy_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same manual-CLI producer path, the nothing-derivable outcome: a
    --summary-file containing the historical content-free placeholder
    records ok=True with required_changes: [] and findings_channel:
    "vacuous" -- never a non-zero exit, matching record_review's
    never-reject invariant (issue #792 AC-3)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text(LEGACY_VACUOUS_SUMMARY, encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "request_changes",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["required_changes"] == []
    assert decision["findings_channel"] == "vacuous"


def test_cli_verdict_isolates_fleet_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #362: cli.main must not write to the operator's real fleet registry.

    Even when no ``--fleet-dir`` is supplied, ``build_app`` calls ``touch_repo``,
    which would otherwise resolve to the global ``%LOCALAPPDATA%`` path. The
    suite-wide autouse fixture redirects writes to a per-test directory; this
    test proves the real default registry is untouched.
    """
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    # Capture the real default fleet path (no env override). The autouse conftest
    # fixture normally sets CHARLIE_WORK_FLEET_DIR, so temporarily clear it to
    # resolve the platform default, then restore the isolated override.
    monkeypatch.delenv("CHARLIE_WORK_FLEET_DIR")
    real_fleet_json = fleet_dir() / "fleet.json"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))

    real_before = real_fleet_json.read_text(encoding="utf-8") if real_fleet_json.exists() else None

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0

    # The write must land in the per-test fleet directory, not the real one.
    isolated_fleet_json = tmp_path / "fleet" / "fleet.json"
    assert isolated_fleet_json.exists()
    data = json.loads(isolated_fleet_json.read_text(encoding="utf-8"))
    assert data["repos"]["owner/repo"]["repo_root"] == str(repo)

    # The operator's real registry must be unchanged (or still absent).
    if real_before is None:
        assert not real_fleet_json.exists()
    else:
        assert real_fleet_json.read_text(encoding="utf-8") == real_before
