"""Tests for the private-slug ratchet CI gate (issue #1502).

Covers the pure scanning functions (``find_slug_mentions_in_diff``,
``count_slug_mentions_in_text``) and the CLI command that wires them
together for ``charlie private-slug-check --base <ref>``.

The tests use synthetic slug names (``secret-repo``, ``hidden-project``)
rather than the real configured slugs, so the test file itself does not
mention the real private slugs and is not counted in the baseline.
"""

from __future__ import annotations

import argparse
import json

import pytest

from charlie_work import cli as cli_module
from charlie_work.config import OrchestratorConfig
from charlie_work.private_slug_gate import (
    SlugMentionDelta,
    SlugMentionFinding,
    count_slug_mentions_in_text,
    find_slug_mentions_in_diff,
)
from charlie_work.subprocess_runner import RunResult

_SLUGS = ["secret-repo", "hidden-project"]


# --- SlugMentionFinding / SlugMentionDelta: frozen dataclass invariant ---


def test_slug_mention_finding_is_frozen() -> None:
    finding = SlugMentionFinding(
        path="src/file.py", line_number=10, slug="secret-repo", content="# todo"
    )
    with pytest.raises(AttributeError):
        finding.path = "other"  # type: ignore[misc]


def test_slug_mention_delta_net_new_property() -> None:
    added = [
        SlugMentionFinding(path="a.py", line_number=1, slug="secret-repo", content="x"),
        SlugMentionFinding(path="b.py", line_number=2, slug="hidden-project", content="y"),
    ]
    removed = [
        SlugMentionFinding(path="c.py", line_number=3, slug="secret-repo", content="z"),
    ]
    delta = SlugMentionDelta(added=added, removed=removed)
    assert delta.net_new == 1


def test_slug_mention_delta_net_new_zero_for_move() -> None:
    added = [SlugMentionFinding(path="a.py", line_number=5, slug="secret-repo", content="# ref")]
    removed = [
        SlugMentionFinding(path="b.py", line_number=10, slug="secret-repo", content="# ref")
    ]
    delta = SlugMentionDelta(added=added, removed=removed)
    assert delta.net_new == 0


def test_slug_mention_delta_is_frozen() -> None:
    delta = SlugMentionDelta(added=[], removed=[])
    with pytest.raises(AttributeError):
        delta.added = []  # type: ignore[misc]


# --- find_slug_mentions_in_diff: core scanning ---


def _make_diff(
    path: str,
    added_lines: list[str],
    removed_lines: list[str] | None = None,
    context: str = "context line",
) -> str:
    """Build a minimal unified diff with the given added and removed lines."""
    removed_lines = removed_lines or []
    lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,{1 + len(removed_lines)} +1,{1 + len(added_lines)} @@",
        f" {context}",
    ]
    for removed in removed_lines:
        lines.append(f"-{removed}")
    for added in added_lines:
        lines.append(f"+{added}")
    lines.append(f" {context}")
    return "\n".join(lines) + "\n"


def test_diff_with_new_slug_mention_is_flagged() -> None:
    diff = _make_diff("src/example.py", ["# see secret-repo for details"])
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 1
    assert delta.removed == []
    assert delta.net_new == 1
    f = delta.added[0]
    assert f.path == "src/example.py"
    assert f.slug == "secret-repo"
    assert f.line_number == 2  # first added line after the context line


def test_diff_with_multiple_new_mentions() -> None:
    diff = _make_diff(
        "src/example.py",
        ["# ref secret-repo here", "# also hidden-project there"],
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 2
    assert delta.net_new == 2
    slugs = {f.slug for f in delta.added}
    assert slugs == {"secret-repo", "hidden-project"}


def test_clean_diff_has_no_findings() -> None:
    diff = _make_diff("src/example.py", ["# no private repos mentioned here"])
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert delta.added == []
    assert delta.removed == []
    assert delta.net_new == 0


def test_move_does_not_produce_net_new() -> None:
    # A refactor that moves a mention: remove from one place, add to another.
    # Net-new = 0, so the gate does not false-positive on moves.
    diff = _make_diff(
        "src/example.py",
        added_lines=["# moved ref to secret-repo"],
        removed_lines=["# old ref to secret-repo"],
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 1
    assert len(delta.removed) == 1
    assert delta.net_new == 0


def test_move_across_files_does_not_produce_net_new() -> None:
    # Move from file A to file B: still net-new = 0 globally.
    diff = (
        "diff --git a/old.py b/old.py\n"
        "--- a/old.py\n"
        "+++ b/old.py\n"
        "@@ -1,2 +1,1 @@\n"
        " # ctx\n"
        "-# ref secret-repo here\n"
        "diff --git a/new.py b/new.py\n"
        "--- a/new.py\n"
        "+++ b/new.py\n"
        "@@ -1,1 +1,2 @@\n"
        " # ctx\n"
        "+# ref secret-repo here\n"
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 1
    assert len(delta.removed) == 1
    assert delta.net_new == 0


def test_removed_mention_counts_as_negative_net_new() -> None:
    diff = _make_diff("src/example.py", added_lines=[], removed_lines=["# ref secret-repo"])
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert delta.added == []
    assert len(delta.removed) == 1
    assert delta.net_new == -1


def test_slug_matched_as_whole_word_only() -> None:
    # "secret-repos" (plural) should NOT match "secret-repo" -- the trailing
    # 's' is a word character, so \b after 'o' fails.
    diff = _make_diff("src/example.py", ["# see secret-repos for details"])
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert delta.added == []


def test_slug_matched_inside_full_repo_path() -> None:
    # "Owner/secret-repo" should match "secret-repo" -- the '/' is a word
    # boundary before 's'.
    diff = _make_diff("src/example.py", ["# see Owner/secret-repo for details"])
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 1
    assert delta.added[0].slug == "secret-repo"


def test_underscore_slug_not_matched_inside_longer_identifier() -> None:
    # Test \b behavior with underscore slugs: '_' is a word character, so
    # no boundary inside a longer identifier.
    slugs = ["fake_runner"]
    # "my_fake_runner" should NOT match -- '_' is a word char, no boundary.
    diff = _make_diff("src/example.py", ["my_fake_runner = []"])
    delta = find_slug_mentions_in_diff(diff, slugs)
    assert delta.added == []
    # "../fake_runner" SHOULD match -- '/' and '.' are non-word.
    diff2 = _make_diff("src/example.py", ["path = ../fake_runner"])
    delta2 = find_slug_mentions_in_diff(diff2, slugs)
    assert len(delta2.added) == 1
    assert delta2.added[0].slug == "fake_runner"


def test_context_lines_are_not_scanned() -> None:
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n"
        " # context secret-repo here\n"
        "-# removed clean line\n"
        "+# added clean line\n"
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert delta.added == []
    assert delta.removed == []


def test_exclude_paths_skips_file() -> None:
    # The baseline file itself should be excluded -- it lists slugs as config.
    diff = _make_diff(".private-slug-baseline.json", ['"slugs": ["secret-repo"]'])
    delta = find_slug_mentions_in_diff(
        diff, _SLUGS, exclude_paths=frozenset({".private-slug-baseline.json"})
    )
    assert delta.added == []
    assert delta.removed == []


def test_empty_diff_has_no_findings() -> None:
    assert find_slug_mentions_in_diff("", _SLUGS).added == []


def test_deleted_file_removed_mentions_counted() -> None:
    # When a file is deleted, removed lines should still be counted (they
    # tighten the ratchet).  +++ /dev/null means new_path is empty, but
    # --- a/path gives old_path for removed-line attribution.
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-# ref secret-repo\n"
        "-# another line\n"
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.removed) == 1
    assert delta.removed[0].path == "gone.py"
    assert delta.removed[0].slug == "secret-repo"


def test_added_line_starting_with_plus_plus_is_scanned() -> None:
    # An added line whose content starts with "++" must not be mistaken for
    # the "+++ b/path" header (same edge case as the mojibake gate).
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,3 @@\n"
        " # ctx\n"
        "+++# ref secret-repo here\n"
        " # ctx\n"
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    assert len(delta.added) == 1
    assert delta.added[0].line_number == 2


def test_line_number_tracking_with_mixed_add_remove() -> None:
    # Verify line numbers stay accurate when added and removed lines
    # interleave in a hunk.
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,4 +1,5 @@\n"
        " # ctx line 1\n"
        "-# old ref secret-repo\n"
        " # ctx line 3\n"
        "+# new ref secret-repo\n"
        "+# also hidden-project\n"
        " # ctx line 5\n"
    )
    delta = find_slug_mentions_in_diff(diff, _SLUGS)
    # New file layout (after applying the hunk):
    #   line 1: "# ctx line 1"     (context)
    #   line 2: "# ctx line 3"     (context, was old line 3 after removing old line 2)
    #   line 3: "# new ref secret-repo"   (added)
    #   line 4: "# also hidden-project"   (added)
    #   line 5: "# ctx line 5"     (context)
    # Old file layout:
    #   line 1: "# ctx line 1"
    #   line 2: "# old ref secret-repo"  (removed)
    #   line 3: "# ctx line 3"
    #   line 4: "# ctx line 5"
    assert len(delta.added) == 2
    assert delta.added[0].line_number == 3
    assert delta.added[1].line_number == 4
    assert len(delta.removed) == 1
    assert delta.removed[0].line_number == 2


# --- count_slug_mentions_in_text ---


def test_count_slug_mentions_in_text() -> None:
    text = (
        "# line 1 mentions secret-repo\n"
        "# line 2 is clean\n"
        "# line 3 mentions hidden-project\n"
        "# line 4 mentions secret-repo again\n"
    )
    assert count_slug_mentions_in_text(text, _SLUGS) == 3


def test_count_slug_mentions_in_text_empty() -> None:
    assert count_slug_mentions_in_text("", _SLUGS) == 0


def test_count_slug_mentions_in_text_no_matches() -> None:
    assert count_slug_mentions_in_text("# nothing here\n# or here\n", _SLUGS) == 0


def test_count_slug_mentions_one_per_line_even_with_two_slugs() -> None:
    # A line mentioning two different slugs counts once (first match wins,
    # consistent with find_slug_mentions_in_diff).
    text = "# secret-repo and hidden-project on the same line\n"
    assert count_slug_mentions_in_text(text, _SLUGS) == 1


# --- CLI wiring: charlie private-slug-check ---


def _cli_args(
    base: str = "origin/main",
    regenerate: bool = False,
    slugs: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=False,
        base=base,
        regenerate=regenerate,
        slugs=slugs,
    )


def _mock_git_diff(stdout: str, ok: bool = True) -> RunResult:
    if ok:
        return RunResult(returncode=0, stdout=stdout, stderr="", error=None)
    return RunResult(returncode=1, stdout="", stderr="boom", error="git diff failed")


def _mock_git_show(stdout: str, ok: bool = True) -> RunResult:
    if ok:
        return RunResult(returncode=0, stdout=stdout, stderr="", error=None)
    return RunResult(returncode=1, stdout="", stderr="not found", error="no such file")


def _write_baseline(tmp_path, *, slugs=None, total=0, files=None) -> None:
    """Write a baseline file for testing."""
    baseline_path = tmp_path / ".private-slug-baseline.json"
    baseline = {
        "version": 1,
        "slugs": _SLUGS if slugs is None else slugs,
        "files": files or {},
        "total": total,
    }
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def _setup_cli_mocks(
    monkeypatch,
    tmp_path,
    *,
    diff_stdout="",
    diff_ok=True,
    base_baseline_total=0,
    base_baseline_exists=True,
) -> None:
    """Wire up the common CLI mocks for private-slug-check tests."""
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())

    def fake_run_captured(command, **kwargs):
        if command[:2] == ["git", "show"]:
            if base_baseline_exists:
                baseline = json.dumps(
                    {"version": 1, "slugs": _SLUGS, "total": base_baseline_total}
                )
                return _mock_git_show(baseline)
            return _mock_git_show("", ok=False)
        if command[:2] == ["git", "diff"]:
            return _mock_git_diff(diff_stdout, ok=diff_ok)
        if command[:2] == ["git", "ls-files"]:
            return RunResult(returncode=0, stdout="", stderr="", error=None)
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)


def test_cli_check_fails_on_new_mention_without_baseline_bump(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, total=10)
    diff = _make_diff("src/new.py", ["# ref secret-repo here"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is False
    assert result.data["net_new"] == 1
    assert result.data["baseline_increase"] == 0
    assert "net-new private-slug mention" in result.message
    assert "src/new.py" in result.message


def test_cli_check_passes_when_baseline_bumped(monkeypatch, tmp_path) -> None:
    # Baseline at HEAD has total=11, at base has total=10 -> increase=1.
    # Diff adds 1 net-new mention.  1 >= 1 -> pass (tamper-evident bump).
    _write_baseline(tmp_path, total=11)
    diff = _make_diff("src/new.py", ["# ref secret-repo here"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is True
    assert result.data["net_new"] == 1
    assert result.data["baseline_increase"] == 1


def test_cli_check_passes_on_clean_diff(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, total=10)
    diff = _make_diff("src/example.py", ["# no slugs here"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args())

    assert result.ok is True
    assert result.data["net_new"] == 0
    assert "clean" in result.message


def test_cli_check_passes_on_move(monkeypatch, tmp_path) -> None:
    # A move: 1 added, 1 removed -> net_new=0 -> pass regardless of baseline.
    _write_baseline(tmp_path, total=10)
    diff = _make_diff(
        "src/example.py",
        added_lines=["# moved ref secret-repo"],
        removed_lines=["# old ref secret-repo"],
    )
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args())

    assert result.ok is True
    assert result.data["net_new"] == 0
    assert result.data["added_count"] == 1
    assert result.data["removed_count"] == 1


def test_cli_check_passes_on_removal(monkeypatch, tmp_path) -> None:
    # Removing a mention: net_new=-1 -> pass (ratchet tightens).
    _write_baseline(tmp_path, total=10)
    diff = _make_diff("src/example.py", added_lines=[], removed_lines=["# ref secret-repo"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args())

    assert result.ok is True
    assert result.data["net_new"] == -1


def test_cli_check_fails_when_baseline_bump_insufficient(monkeypatch, tmp_path) -> None:
    # 2 net-new mentions but baseline only bumped by 1 -> fail.
    _write_baseline(tmp_path, total=11)
    diff = _make_diff("src/new.py", ["# ref secret-repo", "# ref hidden-project"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is False
    assert result.data["net_new"] == 2
    assert result.data["baseline_increase"] == 1


def test_cli_check_passes_when_baseline_file_new_at_head(monkeypatch, tmp_path) -> None:
    # First PR: baseline file doesn't exist at base (base_total=0).
    # Baseline at HEAD has total=5, diff adds 3 net-new, bump=5 >= 3 -> pass.
    _write_baseline(tmp_path, total=5)
    diff = _make_diff(
        "src/new.py", ["# ref secret-repo", "# ref hidden-project", "# ref secret-repo again"]
    )
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_exists=False)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is True
    assert result.data["net_new"] == 3
    assert result.data["baseline_increase"] == 5


def test_cli_check_fails_when_baseline_file_new_but_no_bump(monkeypatch, tmp_path) -> None:
    # First PR: baseline at HEAD has total=0, diff adds 1 net-new -> fail.
    # (This would mean the PR adds mentions but the baseline total is 0.)
    _write_baseline(tmp_path, total=0)
    diff = _make_diff("src/new.py", ["# ref secret-repo"])
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_exists=False)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is False
    assert result.data["net_new"] == 1
    assert result.data["baseline_increase"] == 0


def test_cli_check_reports_git_diff_failure(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, total=10)
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout="", diff_ok=False)

    result = cli_module.run_private_slug_check_command(_cli_args("bad-ref"))

    assert result.ok is False
    assert "could not run git diff" in result.message
    assert result.data["base"] == "bad-ref"


def test_cli_check_uses_two_dot_diff(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, total=10)
    captured: list[list[str]] = []

    def fake_run_captured(command, **kwargs):
        if command[:2] == ["git", "show"]:
            return _mock_git_show(json.dumps({"total": 10}))
        if command[:2] == ["git", "diff"]:
            captured.append(command)
            return _mock_git_diff("")
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)

    cli_module.run_private_slug_check_command(_cli_args("b4d1bf3c"))

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd == ["git", "diff", "b4d1bf3c..HEAD"]
    assert "..." not in cmd[2], f"three-dot diff would fail in shallow clones; got {cmd[2]!r}"


def test_cli_check_excludes_baseline_file_from_scan(monkeypatch, tmp_path) -> None:
    # The baseline file itself is excluded from the diff scan -- it lists
    # slugs as config, not as mentions of the private repos.
    _write_baseline(tmp_path, total=10)
    # Diff includes a change to the baseline file (adding a slug to the list)
    # AND a clean change to a source file.  The baseline file's slug strings
    # should NOT be counted as mentions.
    diff = (
        "diff --git a/.private-slug-baseline.json b/.private-slug-baseline.json\n"
        "--- a/.private-slug-baseline.json\n"
        "+++ b/.private-slug-baseline.json\n"
        "@@ -1,3 +1,3 @@\n"
        " {\n"
        '-  "total": 10,\n'
        '+  "total": 11,\n'
        '   "slugs": ["secret-repo"]\n'
        "diff --git a/src/clean.py b/src/clean.py\n"
        "--- a/src/clean.py\n"
        "+++ b/src/clean.py\n"
        "@@ -1,1 +1,2 @@\n"
        " # ctx\n"
        "+# no slugs here\n"
    )
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout=diff, base_baseline_total=10)

    result = cli_module.run_private_slug_check_command(_cli_args("abc123"))

    assert result.ok is True
    assert result.data["added_count"] == 0
    assert result.data["net_new"] == 0


def test_cli_check_fails_on_missing_baseline_file(monkeypatch, tmp_path) -> None:
    # No baseline file at all -> ConfigError, caught by top-level handler.
    # The command itself raises ConfigError; the test verifies it propagates.
    from charlie_work.config import ConfigError

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())

    with pytest.raises(ConfigError, match="baseline file not found"):
        cli_module.run_private_slug_check_command(_cli_args())


def test_cli_check_fails_on_empty_slugs_list(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, slugs=[], total=0)
    _setup_cli_mocks(monkeypatch, tmp_path, diff_stdout="")

    result = cli_module.run_private_slug_check_command(_cli_args())

    assert result.ok is False
    assert "empty 'slugs'" in result.message


# --- CLI: --regenerate mode ---


def test_cli_regenerate_with_existing_baseline(monkeypatch, tmp_path) -> None:
    _write_baseline(tmp_path, total=10)
    # Simulate git ls-files returning a few files, one of which mentions a slug.
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())

    def fake_run_captured(command, **kwargs):
        if command[:2] == ["git", "ls-files"]:
            return RunResult(
                returncode=0,
                stdout="src/a.py\nsrc/b.py\n.private-slug-baseline.json\n",
                stderr="",
                error=None,
            )
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("# ref secret-repo\n# clean\n", encoding="utf-8")
    (src_dir / "b.py").write_text("# no slugs\n", encoding="utf-8")

    result = cli_module.run_private_slug_check_command(_cli_args(regenerate=True))

    assert result.ok is True
    assert result.data["total"] == 1
    assert result.data["files"] == {"src/a.py": 1}

    # Verify the baseline file was written atomically.
    written = json.loads((tmp_path / ".private-slug-baseline.json").read_text(encoding="utf-8"))
    assert written["total"] == 1
    assert written["files"] == {"src/a.py": 1}
    assert written["slugs"] == _SLUGS


def test_cli_regenerate_with_slugs_arg_when_no_baseline(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())

    def fake_run_captured(command, **kwargs):
        if command[:2] == ["git", "ls-files"]:
            return RunResult(returncode=0, stdout="src/a.py\n", stderr="", error=None)
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("# ref secret-repo\n", encoding="utf-8")

    result = cli_module.run_private_slug_check_command(
        _cli_args(regenerate=True, slugs="secret-repo,hidden-project")
    )

    assert result.ok is True
    assert result.data["total"] == 1
    assert result.data["slugs"] == _SLUGS


def test_cli_regenerate_without_slugs_and_no_baseline_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(
        cli_module,
        "run_captured",
        lambda *a, **k: RunResult(returncode=0, stdout="", stderr="", error=None),
    )

    result = cli_module.run_private_slug_check_command(_cli_args(regenerate=True, slugs=None))

    assert result.ok is False
    assert "--slugs" in result.message
