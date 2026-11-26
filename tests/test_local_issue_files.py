"""Tests for local_issue_files.py: parse/scan/rewrite of markdown issue files.

Tests that exercise malformed files assert on ``scan_issues``'s return value;
what a problem *costs* a pass is ``LocalFileGitHub._handle_scan_problems``'s
policy, tested in test_local_issues.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from charlie_work.local_issue_files import (
    COMMENTS_MARKER,
    STATE_CLOSED,
    STATE_OPEN,
    IssueFileError,
    append_comment,
    issue_number_from_name,
    parse_issue_text,
    render_flow_list,
    rewrite_frontmatter_key,
    scan_issues,
    write_text_atomic,
)


def _issue_text(
    *,
    title: str = "",
    state: str = "",
    labels: str = "",
    created: str = "",
    body: str = "Body text.",
) -> str:
    content: list[str] = []
    if title:
        content.append(f'title: "{title}"')
    if state:
        content.append(f"state: {state}")
    if labels:
        content.append(f"labels: {labels}")
    if created:
        content.append(f"created: {created}")
    if not content:
        # The frontmatter fence regex requires a (possibly blank) line
        # between the two "---" markers; an empty frontmatter block still
        # needs that blank line to be present.
        content.append("")
    lines = ["---", *content, "---", body]
    return "\n".join(lines) + "\n"


def _path(tmp_path: Path, name: str = "001_2026-09-17_a.md") -> Path:
    return tmp_path / name


# -- 1. Filename selection ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("README.md", None),
        ("_template.md", None),
        ("001_x.md.tmp", None),
        ("notes.md", None),
        ("001_2026-09-17_a.md", 1),
        ("12_b.md", 12),
    ],
)
def test_issue_number_from_name(name: str, expected: int | None) -> None:
    assert issue_number_from_name(name) == expected


def test_scan_issues_skips_a_directory_named_like_an_issue_file(tmp_path: Path) -> None:
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "002_dir.md").mkdir()  # a directory, not a file
    (issues_dir / "001_real.md").write_text(_issue_text(title="Real"), encoding="utf-8")

    scan = scan_issues(issues_dir)

    assert [i.number for i in scan.issues] == [1]
    assert scan.problems == ()


# -- 2. Parse -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels_yaml", "expected"),
    [
        ("[bug, automated-ready]", ("bug", "automated-ready")),
        ("[bug, agent:queued]", ("bug", "agent:queued")),  # unquoted-colon form
        ("'agent:queued'", ("agent:queued",)),
        ("[]", ()),
    ],
)
def test_parse_labels_flow_and_scalar_forms(
    tmp_path: Path, labels_yaml: str, expected: tuple[str, ...]
) -> None:
    issue = parse_issue_text(_issue_text(labels=labels_yaml), _path(tmp_path))
    assert issue.labels == expected


def test_parse_labels_block_style(tmp_path: Path) -> None:
    text = "---\nlabels:\n  - bug\n  - automated-ready\n---\nBody.\n"
    issue = parse_issue_text(text, _path(tmp_path))
    assert issue.labels == ("bug", "automated-ready")


def test_parse_labels_missing_key_defaults_empty(tmp_path: Path) -> None:
    issue = parse_issue_text("---\ntitle: x\n---\nBody.\n", _path(tmp_path))
    assert issue.labels == ()


def test_parse_labels_dedupes_order_preserving(tmp_path: Path) -> None:
    text = _issue_text(labels="[bug, automated-ready, bug]")
    issue = parse_issue_text(text, _path(tmp_path))
    assert issue.labels == ("bug", "automated-ready")


@pytest.mark.parametrize(
    ("state_yaml", "expected"),
    [("", STATE_OPEN), ("open", STATE_OPEN), ("closed", STATE_CLOSED)],
)
def test_parse_state_default_and_casing(tmp_path: Path, state_yaml: str, expected: str) -> None:
    issue = parse_issue_text(_issue_text(state=state_yaml), _path(tmp_path))
    assert issue.state == expected


@pytest.mark.parametrize("created_yaml", ['"2026-09-17"', "2026-09-17"])
def test_parse_created_normalizes_to_iso_timestamp(tmp_path: Path, created_yaml: str) -> None:
    issue = parse_issue_text(_issue_text(created=created_yaml), _path(tmp_path))
    assert issue.created_at == "2026-09-17T00:00:00Z"


def test_parse_title_falls_back_to_file_stem(tmp_path: Path) -> None:
    path = _path(tmp_path, "007_2026-09-17_widget.md")
    issue = parse_issue_text("---\nstate: open\n---\nBody.\n", path)
    assert issue.title == "007_2026-09-17_widget"


# -- 3. Parse errors ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("no frontmatter here at all\n", "no '---' fenced frontmatter block"),
        ("---\nlabels: [bug\n---\nBody.\n", "not valid YAML"),
        ("---\n- a\n- b\n---\nBody.\n", "must be a YAML mapping"),
        ("---\nstate: wontfix\n---\nBody.\n", "'state' must be open or closed"),
        ("---\nlabels: {a: 1}\n---\nBody.\n", "'labels' must be a list of strings"),
        ("---\nlabels: [bug, 5]\n---\nBody.\n", "'labels' must be a list of strings"),
    ],
)
def test_parse_errors(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(IssueFileError, match=re.escape(match)):
        parse_issue_text(text, _path(tmp_path))


# -- 4. scan_issues -----------------------------------------------------------


def test_scan_issues_excludes_malformed_file_but_keeps_good_ones(tmp_path: Path) -> None:
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "001_good.md").write_text(_issue_text(title="Good"), encoding="utf-8")
    (issues_dir / "002_bad.md").write_text("no frontmatter\n", encoding="utf-8")

    scan = scan_issues(issues_dir)

    assert [i.number for i in scan.issues] == [1]
    assert len(scan.problems) == 1
    assert scan.problems[0].path == issues_dir / "002_bad.md"


def test_scan_issues_duplicate_number_excludes_and_reports_both(tmp_path: Path) -> None:
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "001_a.md").write_text(_issue_text(title="A"), encoding="utf-8")
    (issues_dir / "001_b.md").write_text(_issue_text(title="B"), encoding="utf-8")

    scan = scan_issues(issues_dir)

    assert scan.issues == ()
    problem_paths = {p.path for p in scan.problems}
    assert problem_paths == {issues_dir / "001_a.md", issues_dir / "001_b.md"}


def test_scan_issues_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    scan = scan_issues(missing)
    assert scan.issues == ()
    assert len(scan.problems) == 1
    assert scan.problems[0].path == missing


def test_scan_issues_is_not_recursive(tmp_path: Path) -> None:
    issues_dir = tmp_path / "issues"
    nested = issues_dir / "sub"
    nested.mkdir(parents=True)
    (nested / "003_nested.md").write_text(_issue_text(title="Nested"), encoding="utf-8")
    (issues_dir / "001_top.md").write_text(_issue_text(title="Top"), encoding="utf-8")

    scan = scan_issues(issues_dir)

    assert [i.number for i in scan.issues] == [1]


# -- 5. rewrite_frontmatter_key -----------------------------------------------


def test_rewrite_frontmatter_key_confines_mutation_to_frontmatter() -> None:
    text = (
        "---\n"
        'title: "fix labels: parsing"\n'
        "state: open\n"
        "labels: [bug]\n"
        "---\n"
        "Some body text.\n"
        "labels: this is prose\n"
        "state: closed\n"
        "More body.\n"
    )
    original_body = text.split("---\n", 2)[2]

    step1 = rewrite_frontmatter_key(text, "labels", "[bug, agent:queued]")
    step2 = rewrite_frontmatter_key(step1, "state", "closed")

    assert step2.split("---\n", 2)[2] == original_body
    title_line = next(line for line in step2.splitlines() if line.startswith("title:"))
    assert title_line == 'title: "fix labels: parsing"'

    # Positive control: a naive whole-file substitution DOES corrupt the body
    # line, proving this fixture is capable of detecting the corruption class
    # (i.e. the confinement assertion above is not vacuous).
    naive = re.sub(r"(labels:).*", r"\1 [x]", text)
    assert naive.split("---\n", 2)[2] != original_body


def test_rewrite_frontmatter_key_collapses_block_style_labels(tmp_path: Path) -> None:
    text = "---\ntitle: t\nlabels:\n  - a\n  - b\ncreated: 2026-01-01\n---\nBody.\n"
    result = rewrite_frontmatter_key(text, "labels", "[a, b, c]")

    lines = result.splitlines()
    assert "labels: [a, b, c]" in lines
    assert not any(line.strip().startswith("- ") for line in lines)
    assert "created: 2026-01-01" in lines

    issue = parse_issue_text(result, _path(tmp_path))
    assert issue.labels == ("a", "b", "c")


def test_rewrite_frontmatter_key_appends_missing_key(tmp_path: Path) -> None:
    text = "---\ntitle: t\nstate: open\n---\nBody.\n"
    result = rewrite_frontmatter_key(text, "resolved", '"2026-09-17"')

    parts = result.split("---\n")
    assert len(parts) == 3
    assert 'resolved: "2026-09-17"' in parts[1]

    issue = parse_issue_text(result, _path(tmp_path))
    assert issue.resolved == "2026-09-17"


def test_rewrite_frontmatter_key_preserves_crlf() -> None:
    text = "---\r\ntitle: t\r\nlabels: [a]\r\nstate: open\r\n---\r\nbody line\r\n"
    result = rewrite_frontmatter_key(text, "labels", "[a, b]")
    data = result.encode("utf-8")
    assert data.count(b"\n") == data.count(b"\r\n")


def test_rewrite_frontmatter_key_preserves_lf() -> None:
    text = "---\ntitle: t\nlabels: [a]\nstate: open\n---\nbody line\n"
    result = rewrite_frontmatter_key(text, "labels", "[a, b]")
    data = result.encode("utf-8")
    assert data.count(b"\r") == 0


def test_rewrite_frontmatter_key_requires_frontmatter() -> None:
    with pytest.raises(IssueFileError):
        rewrite_frontmatter_key("no frontmatter\n", "labels", "[]")


_ROUNDTRIP_LABEL_CASES: list[tuple[str, ...]] = [
    ("agent:queued",),
    ("needs: triage",),
    ("a,b",),
    ("[x]",),
    ("#1",),
    tuple(f"label-{i:02d}" for i in range(30)),
    ("größe",),
]


@pytest.mark.parametrize("labels", _ROUNDTRIP_LABEL_CASES)
def test_rewrite_and_parse_labels_round_trip(tmp_path: Path, labels: tuple[str, ...]) -> None:
    text = _issue_text(labels="[]")
    rewritten = rewrite_frontmatter_key(text, "labels", render_flow_list(labels))

    labels_line = next(line for line in rewritten.splitlines() if line.startswith("labels:"))
    assert "\n" not in labels_line

    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.labels == labels


# -- 6. append_comment ---------------------------------------------------------


def test_append_comment_preserves_body_and_appends_once_then_twice(tmp_path: Path) -> None:
    text = _issue_text(title="T", body="Original body.")
    path = _path(tmp_path)

    once = append_comment(text, "first comment", timestamp="2026-09-17T00:00:00Z")
    issue1 = parse_issue_text(once, path)
    assert issue1.body == "Original body."
    assert issue1.comments == ("**charlie-work** (2026-09-17T00:00:00Z)\n\nfirst comment",)
    assert once.count(COMMENTS_MARKER) == 1

    twice = append_comment(once, "second comment", timestamp="2026-09-17T01:00:00Z")
    issue2 = parse_issue_text(twice, path)
    assert issue2.body == "Original body."
    assert len(issue2.comments) == 2
    assert issue2.comments[1] == "**charlie-work** (2026-09-17T01:00:00Z)\n\nsecond comment"
    assert twice.count(COMMENTS_MARKER) == 1


# -- 7. write_text_atomic -------------------------------------------------------


def test_write_text_atomic_writes_bytes_verbatim_and_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "001_x.md"
    content = "line one\r\nline two\n"

    write_text_atomic(path, content)

    assert path.read_bytes() == content.encode("utf-8")
    assert list(tmp_path.glob("*.tmp")) == []
