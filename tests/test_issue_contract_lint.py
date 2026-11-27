"""Tests for the issue-body contract lint (issue #3115).

``charlie_work.issue_contract_lint.lint_issue_body`` is the in-tree home of the
dispatchable-issue-body rules, delegating every check to the parser the
dispatcher already runs; ``charlie lint-issue`` is its CLI surface (exit 0 all
clean, 1 a body fails, 2 the repo's fleet config cannot be loaded).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work import cli, issue_linking
from charlie_work.issue_contract_lint import lint_issue_body, render_blocker_section


def _fake_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    return repo_root


# --- lint_issue_body: blocker-set rule ----------------------------------------


def test_lint_issue_body_clean_with_matching_blockers() -> None:
    body = "Implement the thing.\n\n## Blocked by\n\nBlocked by #7\n"
    assert lint_issue_body(body, frozenset({7})) == ()


def test_lint_issue_body_clean_empty_body_no_blockers() -> None:
    assert lint_issue_body("Plain prose with no issue references.", frozenset()) == ()


def test_lint_issue_body_blocker_mismatch_reports_both_sets() -> None:
    body = "## Blocked by\n\nBlocked by #3\nBlocked by #5\n"
    (problem,) = lint_issue_body(body, frozenset({5}))
    assert "[3, 5]" in problem
    assert "[5]" in problem


def test_lint_issue_body_missing_expected_blocker_fails() -> None:
    body = "Some body.\n\n## Blocked by\n\nNone\n"
    problems = lint_issue_body(body, frozenset({9}))
    assert len(problems) == 1
    assert "[]" in problems[0] and "[9]" in problems[0]


def test_lint_issue_body_unexpected_blocker_fails() -> None:
    body = "Blocked by #12"
    problems = lint_issue_body(body, frozenset())
    assert len(problems) == 1
    assert "[12]" in problems[0] and "[]" in problems[0]


# --- lint_issue_body: prose-only dependency rule -------------------------------


def test_lint_issue_body_prose_only_dependency_fails() -> None:
    body = "Do not dispatch before P2-T2/P2-T3 have landed."
    (problem,) = lint_issue_body(body, frozenset())
    assert "prose" in problem


def test_lint_issue_body_prose_ordering_ref_fails() -> None:
    body = "Wait for #12 to land first."
    problems = lint_issue_body(body, frozenset())
    assert len(problems) == 1
    assert "prose" in problems[0]


# --- lint_issue_body: path-and-line citation rule ------------------------------


def test_lint_issue_body_path_and_line_citation_fails() -> None:
    body = "The bug is in workflow.py:123 and also config.py:40-45."
    problems = lint_issue_body(body, frozenset())
    assert len(problems) == 2
    assert any("workflow.py:123" in p for p in problems)
    assert any("config.py:40-45" in p for p in problems)


def test_lint_issue_body_symbol_citation_is_clean() -> None:
    body = "The bug is in `parse_blockers` in github_body_scan.py."
    assert lint_issue_body(body, frozenset()) == ()


# --- lint_issue_body: closing-keyword rule --------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Closes #5",
        "closes #5",
        "Closed #5",
        "Fix #5",
        "Fixes #5",
        "fixed #5",
        "Resolve #5",
        "Resolves #5",
        "resolved #5",
        "Closes: #5",
        "Closes:#5",
        "fixes:  #5",
        "This body ends with resolves #42.",
        # Qualified owner/repo#N forms — closing_reference.py's _CLOSING_LINE_RE
        # treats these as closing refs, and GitHub's auto-close honors them.
        "Fixes Senkichi/charlie-work#12",
        "Closes: owner/repo#3",
        "Closes owner/repo#3",
        "resolves some-org/some.repo#9",
        # Hard-wrapped forms — the dispatcher's _CLOSING_KEYWORD_REF binds these
        # through `\s+`, so the lint must flag them too: it may never be
        # narrower than the dispatcher on what counts as a closing ref.
        "this fixes\n#5 as well",
        "fixes\n#5",
        "fix\n\n#5",
        "closes \n #5",
        "Closes:\n#5",
        "Fixes\nowner/repo#3",
    ],
)
def test_lint_issue_body_closing_keyword_fails(body: str) -> None:
    problems = lint_issue_body(body, frozenset())
    assert len(problems) == 1
    assert "closing" in problems[0] or "#5" in problems[0] or "#42" in problems[0]


def test_lint_issue_body_closing_keyword_never_narrower_than_dispatcher() -> None:
    """Dispatcher parity: every whitespace shape ``issue_linking.
    _CLOSING_KEYWORD_REF`` binds — including the hard-wrapped forms its ``\\s+``
    spans across newlines — must also be flagged by the lint. The lint may
    forbid MORE than the dispatcher binds (colon forms, qualified refs, negated
    phrasing) but never less; a ``[ \\t]``-style class that drops ``\\n``
    silently lets a dispatcher-bound body lint clean."""
    for body in (
        "Closes #5",
        "Closes\t#5",
        "resolved  #5",
        "fixes\n#5",
        "fix\n\n#5",
        "closes \n #5",
        "this fixes\n#5 as well",
    ):
        assert issue_linking._CLOSING_KEYWORD_REF.search(body), body
        problems = lint_issue_body(body, frozenset())
        assert any("closing" in p for p in problems), (body, problems)


@pytest.mark.parametrize(
    "body",
    [
        # `\b`-boundary negatives: a keyword occurring only as a word-internal
        # substring ("prefix" contains "fix", "prefixes" contains "fixes")
        # must not trip the rule — no word boundary precedes it.
        "prefix #5",
        "prefixes #5",
        "prefix issue 5",
        # A qualifier that never reaches a #N reference is not a closing ref
        # (mirrors _CLOSING_LINE_RE, which requires the qualifier adjacent to #).
        "Fixes owner/repo ordering.",
    ],
)
def test_lint_issue_body_closing_keyword_boundary_stays_clean(body: str) -> None:
    assert lint_issue_body(body, frozenset()) == ()


def test_lint_issue_body_closing_keyword_negated_still_fails() -> None:
    """The contract forbids the syntax entirely: GitHub's auto-close-on-merge
    has no negation awareness (issue #790), so even a negated closing keyword
    is a violation."""
    problems = lint_issue_body("This does not fix #649.", frozenset())
    assert len(problems) == 1


def test_lint_issue_body_keyword_without_issue_ref_is_clean() -> None:
    body = "Fix the ordering bug in the sweep."
    assert lint_issue_body(body, frozenset()) == ()


# --- issue reference in ordinary prose (acceptance criterion) -------------------


def test_lint_issue_body_issue_ref_in_prose() -> None:
    """An issue reference in ordinary prose with no blocker section fails when
    blockers were expected and passes when none were."""
    body = "See issue #5 for the motivation behind this change."
    assert lint_issue_body(body, frozenset({5})) != ()
    assert lint_issue_body(body, frozenset()) == ()


# --- render_blocker_section -----------------------------------------------------


def test_render_blocker_section_none() -> None:
    assert render_blocker_section([]) == "## Blocked by\n\nNone\n"


def test_render_blocker_section_orders_ascending_one_per_line() -> None:
    assert render_blocker_section([9, 2, 9, 5]) == (
        "## Blocked by\n\nBlocked by #2\nBlocked by #5\nBlocked by #9\n"
    )


def test_render_blocker_section_round_trips_clean() -> None:
    """The rendered section appended to a clean body lints clean against the
    same blocker set (and the dispatcher parses exactly that set)."""
    body = "Add the in-tree lint module.\n\n"
    for blockers in (frozenset(), frozenset({3112}), frozenset({3, 1, 2})):
        rendered = body + render_blocker_section(blockers)
        assert lint_issue_body(rendered, blockers) == ()
        from charlie_work.github_body_scan import parse_blockers

        assert parse_blockers(rendered) == sorted(blockers)


# --- CLI: charlie lint-issue -----------------------------------------------------


def _lint_issue_args(repo: Path, *pairs: tuple[str, str]) -> list[str]:
    argv = ["--repo", str(repo), "lint-issue"]
    for body, blockers in pairs:
        argv += ["--body", body, "--blockers", blockers]
    return argv


def test_lint_issue_cli_clean_exits_0(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n\n## Blocked by\n\nBlocked by #7\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(body), "7"))) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is True
    assert document["ready_label"] == "automated-ready"
    assert document["results"][0]["ok"] is True
    assert document["results"][0]["blockers"] == [7]
    assert document["results"][0]["expected_blockers"] == [7]
    assert document["results"][0]["problems"] == []


def test_lint_issue_cli_failing_body_exits_1(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    good = tmp_path / "good.md"
    good.write_text("Clean body.\n\n## Blocked by\n\nNone\n", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("Closes #5\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(good), ""), (str(bad), ""))) == 1

    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    results = {r["body"]: r for r in document["results"]}
    assert results[str(good)]["ok"] is True
    assert results[str(bad)]["ok"] is False
    assert results[str(bad)]["problems"]


def test_lint_issue_cli_pairs_blockers_positionally(tmp_path: Path, capsys) -> None:
    """The Nth --blockers spec must bind to the Nth --body, not any other.

    Two bodies with DIFFERENT declared blocker sets: A declares #7, B declares
    none. Correct pairing exits 0; swapping the specs must fail BOTH bodies,
    each failure attributed to the correct body. A "first-spec-for-all" or
    "swapped-zip" mutant fails the first invocation.
    """
    repo = _fake_repo(tmp_path)
    body_a = tmp_path / "a.md"
    body_a.write_text("Body A.\n\n## Blocked by\n\nBlocked by #7\n", encoding="utf-8")
    body_b = tmp_path / "b.md"
    body_b.write_text("Body B.\n\n## Blocked by\n\nNone\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(body_a), "7"), (str(body_b), ""))) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is True
    results = {r["body"]: r for r in document["results"]}
    assert results[str(body_a)]["ok"] is True
    assert results[str(body_a)]["expected_blockers"] == [7]
    assert results[str(body_b)]["ok"] is True
    assert results[str(body_b)]["expected_blockers"] == []

    assert cli.main(_lint_issue_args(repo, (str(body_a), ""), (str(body_b), "7"))) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    results = {r["body"]: r for r in document["results"]}
    assert results[str(body_a)]["ok"] is False
    assert results[str(body_a)]["expected_blockers"] == []
    assert results[str(body_a)]["blockers"] == [7]
    assert results[str(body_b)]["ok"] is False
    assert results[str(body_b)]["expected_blockers"] == [7]
    assert results[str(body_b)]["blockers"] == []


def test_lint_issue_cli_missing_body_file_exits_1(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    missing = tmp_path / "missing.md"

    assert cli.main(_lint_issue_args(repo, (str(missing), ""))) == 1

    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    assert document["results"][0]["ok"] is False
    assert document["results"][0]["problems"]


def test_lint_issue_cli_unloadable_config_exits_2(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    (repo / "orchestrator.config.yaml").write_text(
        "not-a-real-section:\n  key: value\n", encoding="utf-8"
    )
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(body), ""))) == 2


def test_lint_issue_cli_ready_label_comes_from_layered_config(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    (repo / "orchestrator.config.yaml").write_text(
        "labels:\n  ready: custom-ready-label\n", encoding="utf-8"
    )
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(body), ""))) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["ready_label"] == "custom-ready-label"


def test_lint_issue_cli_mismatched_pairs_exits_2(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n", encoding="utf-8")

    argv = ["--repo", str(repo), "lint-issue", "--body", str(body), "--blockers", ""]
    argv += ["--body", str(body)]
    assert cli.main(argv) == 2


def test_lint_issue_cli_bad_blocker_csv_exits_2(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n", encoding="utf-8")

    assert cli.main(_lint_issue_args(repo, (str(body), "1,abc"))) == 2


def test_lint_issue_cli_json_flag_wraps_document(tmp_path: Path, capsys) -> None:
    repo = _fake_repo(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Clean body.\n", encoding="utf-8")

    assert cli.main(["--json"] + _lint_issue_args(repo, (str(body), ""))) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is True
    assert document["data"]["ready_label"] == "automated-ready"
    assert document["data"]["results"][0]["ok"] is True
