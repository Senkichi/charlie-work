"""Lint checks that doctor keeps gh-payload field lists constant-driven.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from _field_list_lint import _find_gh_field_list_violations
from _script_loader import load_script_module


def test_gh_field_lists_use_constants_no_inline_literals() -> None:
    """All gh --json field lists must use module-level constants, not inline literals.

    This test scans src/charlie_work/*.py for gh.run() calls with --json arguments
    and verifies that the field list value (the argument after --json) references a
    constant from github.py rather than a string literal. This prevents the contract
    drift issue described in #64 by ensuring all field lists are centralized and not
    scattered as inline strings.

    The matcher recognises three call shapes (issue #1609 added the third):

    1. separate positional string arguments:
       ``gh.run("pr", "list", "--json", "number")``
    2. an ``args=`` keyword whose value is a list:
       ``gh.run(args=["pr", "list", "--json", "number"])``
    3. a single positional list argument:
       ``gh.run(["pr", "list", "--json", "number"], json_output=True)``
    """
    import charlie_work.github as github_module

    # Get the expected constant names from github.py
    expected_constants = {
        "ISSUE_LIST_FIELDS",
        "ISSUE_VIEW_FIELDS",
        "PR_LIST_FIELDS",
        "PR_VIEW_FIELDS",
        "PR_CHECKS_FIELDS",
        "LABEL_LIST_FIELDS",
        "RECONCILE_PR_FIELDS",
        "RECONCILE_ISSUE_FIELDS",
    }

    # Verify constants exist
    for const in expected_constants:
        assert hasattr(github_module, const), f"Missing constant: {const}"
        value = getattr(github_module, const)
        assert isinstance(value, str), f"{const} must be a string"
        assert value, f"{const} must not be empty"

    # Scan all Python files in src/charlie_work/, recursively (Track 2 issue
    # #1588: github_capabilities/ now holds relocated field-list constants
    # and moved method bodies alongside them, e.g. PR_CHECKS_FIELDS in
    # github_capabilities/checks.py -- a non-recursive glob left that whole
    # subpackage, and any other src/charlie_work subpackage, unscanned).
    src_dir = Path(__file__).parent.parent / "src" / "charlie_work"
    violations: list[tuple[str, int, str]] = []

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "github.py":
            # Constant definitions are allowed in github.py
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError):
            continue

        violations.extend(_find_gh_field_list_violations(tree, py_file))

    if violations:
        violation_msg = "\n".join(
            f"  {file}:{line}: {repr(literal)}" for file, line, literal in violations
        )
        raise AssertionError(
            f"Found {len(violations)} field list string literal(s) in gh.run() calls that should use constants:\n"
            f"{violation_msg}\n"
            f"Use the constants from github.py instead (e.g., ISSUE_LIST_FIELDS)."
        )


def test_fake_github_payloads_align_with_field_constants() -> None:
    """FakeGitHub test double payloads must not have keys outside the field constants.

    This prevents the third instance of the #64 bug class: a fake growing a key
    the real gh CLI never returns, which lets dead code pass CI and crash live.
    The dispatch_rework KeyError hotfix on main was caused by exactly this.
    """
    import charlie_work.github as github_module

    # Get the field sets from the constants
    pr_list_fields = set(github_module.PR_LIST_FIELDS.split(","))
    issue_list_fields = set(github_module.ISSUE_LIST_FIELDS.split(","))
    reconcile_pr_fields = set(github_module.RECONCILE_PR_FIELDS.split(","))
    reconcile_issue_fields = set(github_module.RECONCILE_ISSUE_FIELDS.split(","))

    # Check the main FakeGitHub in test_charlie_work.py
    # Import dynamically to avoid module import issues
    test_charlie_work_path = Path(__file__).parent / "test_charlie_work.py"
    test_charlie_work = load_script_module(test_charlie_work_path, "test_charlie_work")

    MainFakeGitHub = test_charlie_work.FakeGitHub
    fake_gh = MainFakeGitHub()

    # Verify PR payload keys are subset of PR_LIST_FIELDS
    pr_keys = set(fake_gh.prs[0].keys())
    extra_pr_keys = pr_keys - pr_list_fields
    assert not extra_pr_keys, (
        f"FakeGitHub.prs[0] has keys not in PR_LIST_FIELDS: {extra_pr_keys}. "
        f"Either remove these keys from the fake or add them to PR_LIST_FIELDS."
    )

    # Verify issue payload keys are subset of ISSUE_LIST_FIELDS
    issue_keys = set(fake_gh.issues[0].keys())
    extra_issue_keys = issue_keys - issue_list_fields
    assert not extra_issue_keys, (
        f"FakeGitHub.issues[0] has keys not in ISSUE_LIST_FIELDS: {extra_issue_keys}. "
        f"Either remove these keys from the fake or add them to ISSUE_LIST_FIELDS."
    )

    # Check the FakeGitHub in test_reconcile.py
    test_reconcile_path = Path(__file__).parent / "test_reconcile.py"
    test_reconcile = load_script_module(test_reconcile_path, "test_reconcile")

    _pr = test_reconcile._pr
    _issue = test_reconcile._issue

    # Verify _pr helper keys are subset of RECONCILE_PR_FIELDS (not PR_LIST_FIELDS)
    # because test_reconcile uses the reconcile field list which includes 'state'
    sample_pr = _pr(1, "OPEN")
    pr_keys = set(sample_pr.keys())
    extra_pr_keys = pr_keys - reconcile_pr_fields
    assert not extra_pr_keys, (
        f"test_reconcile._pr has keys not in RECONCILE_PR_FIELDS: {extra_pr_keys}. "
        f"Either remove these keys from the fake or add them to RECONCILE_PR_FIELDS."
    )

    # Verify _issue helper keys are subset of RECONCILE_ISSUE_FIELDS
    sample_issue = _issue(1, [])
    issue_keys = set(sample_issue.keys())
    extra_issue_keys = issue_keys - reconcile_issue_fields
    assert not extra_issue_keys, (
        f"test_reconcile._issue has keys not in RECONCILE_ISSUE_FIELDS: {extra_issue_keys}. "
        f"Either remove these keys from the fake or add them to RECONCILE_ISSUE_FIELDS."
    )
