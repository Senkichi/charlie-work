"""Tests for ``LocalFileGitHub.create_issue`` / ``edit_issue_body`` (issue #3116).

The local-file backend's half of the two write requests the foreman's Apply
step needs (ADR-0010): creating writes a new ``NNN_<slug>.md`` issue file
with the next free number; editing rewrites only the body. Both pass
title/body through ``outbound_body_guard`` first, and both return errors as
values -- ``None`` / ``False`` -- never raising on a write failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.github import GitHubError
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub


def _write_issue(
    issues_dir: Path,
    number: int,
    *,
    slug: str = "issue",
    title: str = "",
    state: str = "open",
    labels: str = "[]",
    body: str = "Body text.",
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_{slug}.md"
    lines = ["---"]
    if title:
        lines.append(f'title: "{title}"')
    lines.append(f"state: {state}")
    lines.append(f"labels: {labels}")
    lines.append("---")
    lines.append(body)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _state_file(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "state.json"


# A 40-character high-entropy alphanumeric run -- distinct characters, so the
# vendored rules' Shannon-entropy gates pass (synthetic; never a real token).
_GHO = "gho_" + ("x7K9pQ2mN4vB8wL1cR3tY6uI0oP5aS2dF4gH9jZq" * 2)[:36]


@pytest.fixture
def issues_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs" / "issues"
    d.mkdir(parents=True)
    return d


# -- create_issue ------------------------------------------------------------


def test_create_issue_round_trips_title_body_and_labels(issues_dir: Path, tmp_path: Path) -> None:
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    number = gh.create_issue("File the thing: urgently", "Body **text**.", labels=["a", "b"])

    assert number == 1
    view = gh.issue_view(1)
    assert view["title"] == "File the thing: urgently"
    assert view["body"] == "Body **text**."
    assert {label["name"] for label in view["labels"]} == {"a", "b"}
    assert view["state"] == "OPEN"


def test_create_issue_assigns_next_free_number(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1)
    _write_issue(issues_dir, 5)
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert gh.create_issue("first", "b") == 6
    assert gh.create_issue("second", "b") == 7
    assert sorted(i["number"] for i in gh.issue_list(state="all")) == [1, 5, 6, 7]


def test_create_issue_never_collides_with_a_broken_file_claim(
    issues_dir: Path, tmp_path: Path
) -> None:
    """A file scan_issues cannot load still claims its number by filename."""
    _write_issue(issues_dir, 1)
    (issues_dir / "002_broken.md").write_text("---\nstate: [x\n---\n", encoding="utf-8")
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert gh.create_issue("new", "b") == 3


def test_create_issue_secret_title_or_body_is_refused(issues_dir: Path, tmp_path: Path) -> None:
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert gh.create_issue("t", f"notes {_GHO}") is None
    assert gh.create_issue(f"wip {_GHO}", "clean") is None
    assert list(issues_dir.iterdir()) == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert [e["payload"]["surface"] for e in events] == ["create_issue", "create_issue"]


def test_create_issue_dry_run_writes_nothing(issues_dir: Path, tmp_path: Path) -> None:
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir, dry_run=True)

    assert gh.create_issue("t", "b") is None
    assert list(issues_dir.iterdir()) == []


def test_create_issue_missing_dir_raises_github_error(tmp_path: Path) -> None:
    """Same config-error contract as every read: GitHubError, never a silent no-op."""
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path / "docs" / "issues")
    with pytest.raises(GitHubError):
        gh.create_issue("t", "b")


# -- edit_issue_body ---------------------------------------------------------


def test_edit_issue_body_changes_the_body_and_nothing_else(
    issues_dir: Path, tmp_path: Path
) -> None:
    path = _write_issue(issues_dir, 3, title="T", labels="[a]", body="old body")
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert gh.edit_issue_body(3, "completely new body") is True

    view = gh.issue_view(3)
    assert view["body"] == "completely new body"
    assert view["title"] == "T"
    assert {label["name"] for label in view["labels"]} == {"a"}
    assert view["state"] == "OPEN"
    # The frontmatter block survived the edit byte-for-byte.
    text = path.read_text(encoding="utf-8")
    frontmatter_end = text.index("---", 4)
    assert "labels: [a]" in text[:frontmatter_end]
    assert 'title: "T"' in text[:frontmatter_end]
    assert text.endswith("completely new body\n")


def test_edit_issue_body_preserves_comments_and_crlf(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 2, body="old body")
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    body_file = tmp_path / "c.md"
    body_file.write_text("a comment\n", encoding="utf-8")
    gh.issue_comment(2, body_file)
    raw = path.read_bytes().decode("utf-8")
    crlf = raw.replace("\r\n", "\n").replace("\n", "\r\n")
    path.write_bytes(crlf.encode("utf-8"))

    assert gh.edit_issue_body(2, "rewritten body") is True

    text = path.read_bytes().decode("utf-8")
    assert "\r\n" in text and "old body" not in text
    assert "rewritten body" in text
    assert "<!-- charlie-work:comments -->" in text
    assert "a comment" in text
    view = gh.issue_view(2)
    assert view["body"] == "rewritten body"
    assert len(view["comments"]) == 1
    assert "a comment" in view["comments"][0]["body"]


def test_edit_issue_body_secret_refused_and_untouched(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 4, body="old body")
    before = path.read_bytes()
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert gh.edit_issue_body(4, f"notes {_GHO}") is False
    assert path.read_bytes() == before
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "edit_issue_body"
    assert events[0]["issue_number"] == 4


def test_edit_issue_body_missing_issue_and_dry_run(issues_dir: Path, tmp_path: Path) -> None:
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    assert gh.edit_issue_body(99, "x") is False

    _write_issue(issues_dir, 1, body="old body")
    dry = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir, dry_run=True)
    assert dry.edit_issue_body(1, "x") is False
    assert gh.issue_view(1)["body"] == "old body"
