"""Tests for local_issues.py: LocalFileGitHub, a GitHubLike over markdown files."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from charlie_work import labels as labels_module
from charlie_work.config import LabelConfig
from charlie_work.github import GitHubError, GitHubLike, GitHubRunResult
from charlie_work.github_capabilities.issues import get_github_issue_dependencies
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


@pytest.fixture
def issues_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs" / "issues"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def gh(tmp_path: Path, issues_dir: Path) -> LocalFileGitHub:
    return LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)


# -- 8. Protocol completeness, derived from GitHubLike -------------------------


def _iter_like_protocols() -> list[type]:
    return [c for c in GitHubLike.__mro__ if c.__name__.endswith("Like")]


def _protocol_members(proto: type) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for name, value in vars(proto).items():
        if name.startswith("_"):
            continue
        if isinstance(value, property) or callable(value):
            members[name] = value
    return members


def _derived_members() -> dict[str, Any]:
    derived: dict[str, Any] = {}
    for proto in _iter_like_protocols():
        derived.update(_protocol_members(proto))
    return derived


def test_derived_protocol_member_set_is_non_vacuous() -> None:
    """Positive control: the collector actually finds a real, sizeable surface."""
    derived = _derived_members()
    assert {"issue_list", "add_issue_label", "pr_list", "merge_pr", "dry_run"} <= set(derived)
    assert len(derived) > 30


def test_local_file_github_defines_every_protocol_member() -> None:
    derived = _derived_members()
    missing = [name for name in derived if not hasattr(LocalFileGitHub, name)]
    assert missing == []


def test_local_file_github_satisfies_githublike_isinstance(tmp_path: Path) -> None:
    instance = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path / "issues")
    assert isinstance(instance, GitHubLike)


def test_local_file_github_method_signatures_match_protocol_keyword_names() -> None:
    """A renamed keyword parameter would break every keyword-argument caller."""
    derived = _derived_members()
    mismatches = []
    for name, proto_member in derived.items():
        if isinstance(proto_member, property):
            continue
        impl_member = getattr(LocalFileGitHub, name)
        proto_params = [p for p in inspect.signature(proto_member).parameters if p != "self"]
        impl_params = [p for p in inspect.signature(impl_member).parameters if p != "self"]
        if impl_params != proto_params:
            mismatches.append((name, proto_params, impl_params))
    assert mismatches == []


# -- 9. issue_list --------------------------------------------------------------


def test_issue_list_label_filter_is_and(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1, labels="[bug]")
    _write_issue(issues_dir, 2, labels="[bug, automated-ready]")
    _write_issue(issues_dir, 3, labels="[automated-ready]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    result = client.issue_list(labels=["bug", "automated-ready"])
    assert [d["number"] for d in result] == [2]


def test_issue_list_accepts_bare_str_label(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1, labels="[bug]")
    _write_issue(issues_dir, 2, labels="[bug, automated-ready]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    result = client.issue_list(labels="bug")
    assert {d["number"] for d in result} == {1, 2}


@pytest.mark.parametrize(
    ("state", "expected_numbers"),
    [("open", {1}), ("closed", {2}), ("all", {1, 2}), (None, {1})],
)
def test_issue_list_state_filter(
    issues_dir: Path, tmp_path: Path, state: str | None, expected_numbers: set[int]
) -> None:
    _write_issue(issues_dir, 1, state="open")
    _write_issue(issues_dir, 2, state="closed")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    result = client.issue_list(state=state)
    assert {d["number"] for d in result} == expected_numbers


def test_issue_list_newest_first(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1)
    _write_issue(issues_dir, 5)
    _write_issue(issues_dir, 3)
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert [d["number"] for d in client.issue_list()] == [5, 3, 1]


def test_issue_list_dict_shape_and_url(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1, slug="widget", title="Widget", labels="[bug]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    [d] = client.issue_list()

    assert set(d.keys()) == {
        "number",
        "title",
        "url",
        "body",
        "labels",
        "state",
        "createdAt",
        "updatedAt",
        "author",
        "comments",
        "assignees",
    }
    assert d["labels"] == [{"name": "bug"}]
    assert d["author"] == {"login": ""}
    assert d["url"] == "local://docs/issues/001_widget.md"


# -- 10. issue_view / are_issues_open -------------------------------------------


def test_issue_view_missing_returns_empty_dict(gh: LocalFileGitHub) -> None:
    assert gh.issue_view(999) == {}


def test_are_issues_open_excludes_closed_and_missing(issues_dir: Path, tmp_path: Path) -> None:
    _write_issue(issues_dir, 1, state="open")
    _write_issue(issues_dir, 2, state="closed")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert client.are_issues_open([1, 2, 999]) == {1}


# -- 11. Real label state machine ------------------------------------------------


def _labels_on_disk(repo_root: Path, issues_dir: Path, number: int) -> set[str]:
    fresh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    view = fresh.issue_view(number)
    return {entry["name"] for entry in view["labels"]}


def test_labels_transition_drives_real_state_machine(issues_dir: Path, tmp_path: Path) -> None:
    cfg = LabelConfig()
    _write_issue(issues_dir, 1, labels=f"[{cfg.ready}, bug]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    steps = [
        ("queued", {cfg.ready, "bug", cfg.queued}),
        ("dispatched", {cfg.ready, "bug", cfg.in_progress}),
        ("escalated", {cfg.ready, "bug", cfg.human_needed}),
    ]
    for event, expected in steps:
        result = labels_module.transition(client, cfg, 1, event)
        assert result.outcome == labels_module.TransitionOutcome.APPLIED
        assert _labels_on_disk(tmp_path, issues_dir, 1) == expected

    result = labels_module.transition(client, cfg, 1, "merged")
    assert result.outcome == labels_module.TransitionOutcome.APPLIED
    final_labels = _labels_on_disk(tmp_path, issues_dir, 1)
    assert cfg.ready not in final_labels
    assert final_labels == {"bug", cfg.done}


# -- 12. Idempotence without writes ----------------------------------------------


def test_add_existing_label_is_idempotent_no_write(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 1, labels="[bug]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    assert client.add_issue_label(1, "bug") is True

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime


def test_remove_absent_label_is_idempotent_no_write(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 1, labels="[bug]")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    assert client.remove_issue_label(1, "not-present") is True

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime


# -- 13. Missing issue number ------------------------------------------------------


def test_missing_issue_number_returns_false_never_raises(gh: LocalFileGitHub) -> None:
    assert gh.add_issue_label(999, "x") is False
    assert gh.remove_issue_label(999, "x") is False
    assert gh.close_issue(999) is False


# -- 14. close_issue -----------------------------------------------------------------


def test_close_issue_sets_state_and_stamps_resolved(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 1, state="open", body="state: open\nMore body.")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert client.close_issue(1) is True

    text = path.read_text(encoding="utf-8")
    fresh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    assert fresh.issue_view(1)["state"] == "CLOSED"
    assert "state: open" in text.split("---\n", 2)[2]  # body line untouched

    today = datetime.now(UTC).date().isoformat()
    assert f'resolved: "{today}"' in text

    bytes_after_first_close = path.read_bytes()
    assert client.close_issue(1) is True
    assert path.read_bytes() == bytes_after_first_close


def test_close_issue_does_not_overwrite_existing_resolved(
    issues_dir: Path, tmp_path: Path
) -> None:
    path = issues_dir / "001_x.md"
    path.write_text(
        '---\ntitle: "t"\nstate: open\nlabels: []\nresolved: "2020-01-01"\n---\nBody.\n',
        encoding="utf-8",
    )
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    assert client.close_issue(1) is True

    assert 'resolved: "2020-01-01"' in path.read_text(encoding="utf-8")


# -- 15. dry_run ----------------------------------------------------------------------


def test_dry_run_add_remove_close_and_comment_are_no_ops(issues_dir: Path, tmp_path: Path) -> None:
    path = _write_issue(issues_dir, 1, labels="[bug]")
    body_file = tmp_path / "comment.txt"
    body_file.write_text("a comment", encoding="utf-8")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir, dry_run=True)
    before = path.read_bytes()

    assert client.add_issue_label(1, "new-label") is True
    assert path.read_bytes() == before

    assert client.remove_issue_label(1, "bug") is True
    assert path.read_bytes() == before

    assert client.close_issue(1) is True
    assert path.read_bytes() == before

    assert client.issue_comment(1, body_file) is None
    assert path.read_bytes() == before


# -- 16. Null-object surface -----------------------------------------------------------


def test_null_object_reads(gh: LocalFileGitHub) -> None:
    assert gh.pr_list() == []
    assert gh.merged_pr_list() == []
    assert gh.pr_view(1) == {}
    assert gh.pr_checks(1) is None
    assert gh.pr_create("head", "base", "title", "body") is None

    result = gh.merged_prs_for_issue(1, "agent/")
    assert list(result) == []
    assert result.ok is True

    sufficient, _remaining, _reset_at = gh.check_graphql_rate_limit()
    assert sufficient is True

    assert gh.delete_branch("x") is False


@pytest.mark.parametrize("method_name", ["pr_ready", "pr_close", "pr_reopen"])
def test_null_object_pr_number_actions_fail_as_values(
    gh: LocalFileGitHub, method_name: str
) -> None:
    result = getattr(gh, method_name)(1)
    assert isinstance(result, GitHubRunResult)
    assert result.ok is False
    assert result.error


def test_null_object_push_empty_commit_fails_as_value(gh: LocalFileGitHub) -> None:
    result = gh.push_empty_commit("some-branch")
    assert isinstance(result, GitHubRunResult)
    assert result.ok is False
    assert result.error


def test_null_object_commit_fails_as_value(gh: LocalFileGitHub) -> None:
    result = gh.commit("deadbeef")
    assert isinstance(result, GitHubRunResult)
    assert result.ok is False
    assert result.error


def test_run_allow_failure_returns_structured_failure(gh: LocalFileGitHub) -> None:
    result = gh.run(["issue", "list"], allow_failure=True)
    assert isinstance(result, GitHubRunResult)
    assert result.ok is False
    assert result.error


def test_run_raises_without_allow_failure(gh: LocalFileGitHub) -> None:
    with pytest.raises(GitHubError):
        gh.run(["issue", "list"])


def test_merge_pr_and_pr_comment_raise(gh: LocalFileGitHub, tmp_path: Path) -> None:
    with pytest.raises(GitHubError):
        gh.merge_pr(1, "squash")

    body_file = tmp_path / "c.txt"
    body_file.write_text("x", encoding="utf-8")
    with pytest.raises(GitHubError):
        gh.pr_comment(1, body_file)


def test_repo_owner_name_raises(gh: LocalFileGitHub) -> None:
    with pytest.raises(GitHubError):
        gh._repo_owner_name()


def test_name_with_owner(tmp_path: Path) -> None:
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path / "issues")
    assert client.name_with_owner() == f"local/{tmp_path.name}"


# -- 17. Warm-cache contract for issue_dependencies ------------------------------------


@dataclass(frozen=True)
class _SpyLocalFileGitHub(LocalFileGitHub):
    """``LocalFileGitHub`` subclass that records every ``run`` invocation.

    The dataclass is frozen, so ``calls`` is its own field (a mutable list
    that gets appended to in place, never reassigned) rather than something
    set via ``object.__setattr__``.
    """

    calls: list[list[str]] = field(default_factory=list, compare=False, repr=False)

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        self.calls.append(list(args))
        return super().run(args, json_output=json_output, allow_failure=allow_failure)


def test_issue_dependencies_warms_cache_and_skips_the_api_call(
    tmp_path: Path, issues_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spy = _SpyLocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    spy.issue_dependencies([1, 2])

    with caplog.at_level(logging.WARNING):
        deps = get_github_issue_dependencies(spy, 1)

    assert deps == []
    assert spy.calls == []
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_issue_dependencies_cache_miss_does_log_a_warning(
    tmp_path: Path, issues_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Positive control: proves the previous test's assertions can fail."""
    spy = _SpyLocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    spy._list_cache.clear()

    with caplog.at_level(logging.WARNING):
        deps = get_github_issue_dependencies(spy, 999)

    assert deps == []
    assert len(spy.calls) == 1
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


# -- 18. Containment --------------------------------------------------------------------


def test_issues_dir_outside_repo_root_raises_relative_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    escaping = repo_root / ".." / "x"
    with pytest.raises(ValueError):
        LocalFileGitHub(repo_root=repo_root, issues_dir=escaping)


def test_issues_dir_outside_repo_root_raises_absolute_sibling(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sibling = tmp_path / "other"
    sibling.mkdir()
    with pytest.raises(ValueError):
        LocalFileGitHub(repo_root=repo_root, issues_dir=sibling)


# -- 19. issue_comment --------------------------------------------------------------------


def test_issue_comment_missing_issue_raises(gh: LocalFileGitHub, tmp_path: Path) -> None:
    body_file = tmp_path / "c.txt"
    body_file.write_text("hello", encoding="utf-8")
    with pytest.raises(GitHubError):
        gh.issue_comment(999, body_file)


def test_issue_comment_appends_and_is_visible_via_issue_view(
    issues_dir: Path, tmp_path: Path
) -> None:
    _write_issue(issues_dir, 1, body="Original body.")
    client = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    body_file = tmp_path / "c.txt"
    body_file.write_text("a real comment", encoding="utf-8")

    client.issue_comment(1, body_file)

    view = client.issue_view(1)
    assert view["body"] == "Original body."
    assert len(view["comments"]) == 1
    assert "a real comment" in view["comments"][0]["body"]


# -- scan-problem policy --------------------------------------------------------


def test_broken_issue_file_warns_once_per_instance_and_pass_continues(
    issues_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_issue(issues_dir, 1, labels="[a]")
    broken = issues_dir / "002_broken.md"
    broken.write_text("---\nstate: [unclosed\n---\nbody\n", encoding="utf-8")
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    with caplog.at_level(logging.WARNING, logger="charlie_work.local_issues"):
        first = gh.issue_list()
        gh.issue_view(1)
        gh.issue_list(labels=["a"])

    assert [i["number"] for i in first] == [1]  # the pass went on without #2
    skipped = [r for r in caplog.records if "local issue file skipped" in r.getMessage()]
    assert len(skipped) == 1, [r.getMessage() for r in skipped]
    assert "002_broken.md" in skipped[0].getMessage()


def test_fresh_instance_warns_again(
    issues_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dedupe is per instance: the fleet loop rebuilds the client each pass,
    so a still-broken file is reported once per pass, never silenced for good."""
    (issues_dir / "002_broken.md").write_text("---\nstate: [x\n---\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.local_issues"):
        LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir).issue_list()
        LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir).issue_list()

    assert sum("local issue file skipped" in r.getMessage() for r in caplog.records) == 2


def test_duplicate_issue_number_warns_and_neither_claimant_is_served(
    issues_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_issue(issues_dir, 1, slug="one")
    _write_issue(issues_dir, 7, slug="a")
    _write_issue(issues_dir, 7, slug="b")
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    with caplog.at_level(logging.WARNING, logger="charlie_work.local_issues"):
        numbers = [i["number"] for i in gh.issue_list()]

    assert numbers == [1]
    assert gh.issue_view(7) == {}
    messages = [
        r.getMessage() for r in caplog.records if "local issue file skipped" in r.getMessage()
    ]
    assert len(messages) == 2 and all("issue number 7" in m for m in messages)


def test_missing_issues_dir_raises_github_error(tmp_path: Path) -> None:
    """No directory is not "zero issues": defer the pass, as a gh outage would."""
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path / "docs" / "issues")

    with pytest.raises(GitHubError, match="issues directory does not exist"):
        gh.issue_list()
    with pytest.raises(GitHubError):
        gh.add_issue_label(1, "x")
