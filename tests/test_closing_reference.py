"""Tests for ``charlie_work.closing_reference`` (cw#1263).

All ``gh`` interaction here is mocked -- ``validate_closing_reference`` never
performs real I/O itself; the optional liveness probe is whatever object the
caller supplies.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from _src_ast import parsed, source_files, source_text

from charlie_work.closing_reference import (
    ValidationResult,
    closing_issues_referenced_numbers,
    validate_closing_reference,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "charlie_work"


class _FakeIssueViewer:
    """Mock ``gh`` satisfying the narrow ``issue_view`` surface."""

    def __init__(self, state: str | None = "OPEN", *, raises: bool = False) -> None:
        self._state = state
        self._raises = raises
        self.calls: list[int] = []

    def issue_view(self, number: int) -> dict[str, Any]:
        self.calls.append(number)
        if self._raises:
            raise RuntimeError("gh unavailable")
        if self._state is None:
            return {}
        return {"number": number, "state": self._state}


def test_wrong_number_is_rewritten_to_canonical() -> None:
    body = "Closes #7\n\nSalvaged by the orchestrator from a worker branch."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert isinstance(result, ValidationResult)
    assert result.changed is True
    assert "Closes #42" in result.body
    assert "Closes #7" not in result.body
    assert result.findings == ("closing reference rewritten",)


def test_missing_closing_line_is_added() -> None:
    body = "Salvaged by the orchestrator from a worker branch."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert result.body.startswith("Closes #42\n\n")
    assert "Salvaged by the orchestrator" in result.body
    assert result.findings == ("missing closing line added",)


def test_missing_closing_line_on_empty_body() -> None:
    result = validate_closing_reference("", 42, "owner/repo")

    assert result.body == "Closes #42"
    assert result.changed is True


def test_correct_bare_reference_is_left_untouched() -> None:
    body = "Closes #42\n\nSalvaged by the orchestrator from a worker branch."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.body == body
    assert result.changed is False
    assert result.findings == ()


def test_correct_cross_repo_form_preserved_untouched() -> None:
    """The whole point of the ``repo`` parameter: a correct ``owner/repo#N``
    qualifier must survive byte-for-byte, never collapsed to bare ``#N``."""
    body = "Closes owner/repo#42\n\nSalvaged by the orchestrator from a worker branch."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.body == body
    assert result.changed is False
    assert "owner/repo#42" in result.body


def test_wrong_cross_repo_qualifier_is_rewritten_with_the_correct_repo() -> None:
    """A qualifier naming the WRONG repo is a defect too, not just a wrong
    number -- the rewrite must substitute the correct ``repo``, not just
    keep echoing the untrusted one back."""
    body = "Closes other/repo#42\n\nSalvaged by the orchestrator from a worker branch."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert "owner/repo#42" in result.body
    assert "other/repo#42" not in result.body


def test_wrong_number_with_cross_repo_qualifier_preserves_qualified_style() -> None:
    body = "Closes owner/repo#7\n\nrest"

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert "Closes owner/repo#42" in result.body


def test_multiple_closing_lines_collapsed_to_one() -> None:
    body = "Closes #1\nFixes #2\n\nrest of body"

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert result.body.count("Closes #42") == 1
    assert "#1" not in result.body
    assert "#2" not in result.body
    assert result.findings == ("multiple closing lines collapsed to one",)


def test_matching_prose_earlier_in_body_does_not_shadow_the_real_line() -> None:
    """Regression: the same literal closing-line text can also occur as an
    unstructured substring earlier in the body (e.g. pulled in from a
    commit-message excerpt via ``summarize_branch_work``). A content-based
    ``str.replace`` finds and corrupts that first occurrence instead of the
    actual structured line, leaving the real (wrong) closing line
    untouched -- the rewrite must be positional, not literal-text-based."""
    body = (
        "Note: someone once wrote Closes #99 here by mistake in prose.\n\nCloses #99\n\nmore text"
    )

    result = validate_closing_reference(body, 6, "owner/repo")

    assert result.changed is True
    # The prose sentence must survive untouched...
    assert "someone once wrote Closes #99 here by mistake in prose." in result.body
    # ...while the real structured line is the one that got corrected.
    assert "Closes #6" in result.body
    lines = result.body.splitlines()
    structured_lines = [ln for ln in lines if ln.strip() in ("Closes #99", "Closes #6")]
    assert structured_lines == ["Closes #6"]


def test_matching_prose_earlier_in_body_does_not_shadow_removed_duplicate() -> None:
    """Same hazard, collapse-branch (rule 4): an extra structured closing
    line must actually be removed even when its exact text also appears as
    prose earlier in the body."""
    body = "Text that mentions Closes #2 in passing.\n\nCloses #1\nCloses #2\n\nrest"

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert "Text that mentions Closes #2 in passing." in result.body
    lines = result.body.splitlines()
    structured_lines = [
        ln for ln in lines if ln.strip() in ("Closes #1", "Closes #2", "Closes #42")
    ]
    assert structured_lines == ["Closes #42"]


def test_fixes_keyword_recognized_same_as_closes() -> None:
    body = "Fixes #42\n\nrest"

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is False


def test_keyword_mid_sentence_is_not_mistaken_for_a_structured_line() -> None:
    """``linked_issue_number``'s keyword scan intentionally matches keywords
    anywhere in prose; this module's write-time validator is narrower by
    design -- it only recognizes a line consisting of nothing but the
    closing reference, so a mid-sentence mention doesn't count as "already
    present" and both the mention and a canonical line coexist."""
    body = "This closes #4 loose ends in the implementation."

    result = validate_closing_reference(body, 42, "owner/repo")

    assert result.changed is True
    assert result.body.startswith("Closes #42\n\n")
    assert "closes #4 loose ends" in result.body


def test_closed_issue_target_is_logged_but_body_still_usable() -> None:
    """A closed target issue is informational only -- it never withholds or
    mutates the body beyond the ordinary correctness rules."""
    body = "Closes #42\n\nrest"
    gh = _FakeIssueViewer(state="CLOSED")

    result = validate_closing_reference(body, 42, "owner/repo", gh=gh)

    assert result.target_issue_open is False
    assert result.body == body
    assert result.changed is False
    assert gh.calls == [42]


def test_open_issue_target_probed() -> None:
    gh = _FakeIssueViewer(state="OPEN")

    result = validate_closing_reference("Closes #42\n\nrest", 42, "owner/repo", gh=gh)

    assert result.target_issue_open is True


def test_no_probe_when_gh_not_supplied() -> None:
    result = validate_closing_reference("Closes #42\n\nrest", 42, "owner/repo")

    assert result.target_issue_open is None


def test_probe_failure_degrades_to_none_without_raising() -> None:
    gh = _FakeIssueViewer(raises=True)

    result = validate_closing_reference("Closes #42\n\nrest", 42, "owner/repo", gh=gh)

    assert result.target_issue_open is None
    assert result.body == "Closes #42\n\nrest"


def test_probe_returning_empty_dict_degrades_to_none() -> None:
    gh = _FakeIssueViewer(state=None)

    result = validate_closing_reference("Closes #42\n\nrest", 42, "owner/repo", gh=gh)

    assert result.target_issue_open is None


def test_never_raises_on_malformed_input() -> None:
    # A body containing regex-metacharacter-heavy text must not blow up the
    # matcher or the string-replace step.
    body = "Closes #1 (?:weird) [brackets] $pecial\nFixes #2\n"

    result = validate_closing_reference(body, 99, "owner/repo")

    assert isinstance(result, ValidationResult)
    assert "Closes #99" in result.body


def test_validation_result_is_frozen() -> None:
    result = validate_closing_reference("Closes #1\n", 1, "owner/repo")
    try:
        result.changed = False  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstanceError" in type(exc).__name__
    else:
        raise AssertionError("ValidationResult must be frozen")


def test_closing_issues_referenced_numbers_parses_graphql_shape() -> None:
    pr_view = {
        "closingIssuesReferences": [
            {"number": 42, "title": "some issue"},
            {"number": 7, "title": "another issue"},
        ]
    }

    assert closing_issues_referenced_numbers(pr_view) == {42, 7}


def test_closing_issues_referenced_numbers_empty_on_missing_field() -> None:
    assert closing_issues_referenced_numbers({}) == set()


def test_closing_issues_referenced_numbers_empty_on_malformed_field() -> None:
    assert closing_issues_referenced_numbers({"closingIssuesReferences": "not-a-list"}) == set()
    assert (
        closing_issues_referenced_numbers(
            {"closingIssuesReferences": [{"number": "not-an-int"}, {"no_number": 1}]}
        )
        == set()
    )


# ---------------------------------------------------------------------------
# AC2: no `gh.pr_create` call site may bypass `validate_closing_reference`.
#
# This derives call sites dynamically from the source tree via an AST scan --
# no hardcoded file list -- so a future third call site (a new salvage lane,
# a new adapter) is caught automatically rather than silently exempted.
# ---------------------------------------------------------------------------


class _FileFacts(NamedTuple):
    """One scanned source file's verdicts, shared by both AC guards below.

    ``has_raw_call`` -- a ``.pr_create(`` attribute call, or a call through a
    name assigned from ``gh.pr_create`` / ``getattr(gh, "pr_create", ...)``,
    outside a ``def pr_create`` body. ``has_wrapper_call`` -- a
    ``create_pr_with_retry(`` call anywhere in the file (cw#1273).
    ``uses_validator`` -- the file imports and calls
    ``validate_closing_reference``; only computed when ``has_wrapper_call``
    is true and left False otherwise, because the validator guard below
    only inspects wrapper consumers.
    """

    has_raw_call: bool
    has_wrapper_call: bool
    uses_validator: bool


def _call_site_facts(tree: ast.AST) -> tuple[bool, bool]:
    """One pass over ``tree``: ``(raw .pr_create call site, create_pr_with_retry call site)``.

    The first flag covers both indirect-call shapes without hardcoding which
    module uses them:
    - ``pr_create = getattr(gh, "pr_create", None); pr_create(...)``
    - ``fn = gh.pr_create; fn(...)`` (an ``ast.Attribute`` bound to a name,
      no ``getattr`` involved)

    cw#1273 interposed ``pr_create_retry.create_pr_with_retry`` between every
    caller and the raw ``gh.pr_create`` -- after that change, the raw
    ``.pr_create(`` attribute call lives *only* inside ``pr_create_retry.py``
    itself (see ``test_every_pr_create_call_site_routes_through_the_retry_wrapper``
    below), so it is no longer where a PR actually gets created from the
    perspective of "did this caller build/validate a body first". The second
    flag follows that indirection the same way the first follows the
    ``getattr(gh, "pr_create", None)`` alias indirection: both track *the
    point closest to the caller* where a PR creation is actually requested.

    A ``def pr_create(...)`` body (or an async equivalent) is not a call site
    of itself, so calls inside it are not collected -- but an alias assigned
    inside one still counts for calls elsewhere and a wrapper call inside one
    still counts, matching the whole-tree reads the original helpers made.
    Alias targets and called names are collected in the same DFS and joined
    after the walk, which keeps the four separate whole-tree ``ast.walk``
    passes (alias collection, the ``def pr_create`` skip-set walk, the
    materialized outside-definition node list, and the wrapper scan) the
    original helpers made per matched file down to one.
    """
    aliases: set[str] = set()
    name_calls: set[str] = set()
    raw_call = False
    wrapper_call = False

    def collect_alias(node: ast.Assign) -> None:
        value = node.value
        is_getattr_alias = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
            and isinstance(value.args[1], ast.Constant)
            and value.args[1].value == "pr_create"
        )
        is_attribute_alias = isinstance(value, ast.Attribute) and value.attr == "pr_create"
        if is_getattr_alias or is_attribute_alias:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)

    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "pr_create":
            # Excludes only the *definition* of ``pr_create`` itself (its own
            # body, where a call to ``self`` or an inner helper is not a call
            # site the validator needs to see), not the whole module -- a
            # retry wrapper or any other call to ``gh.pr_create`` living
            # elsewhere in the same file must still be caught.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign):
                    collect_alias(inner)
                elif (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "create_pr_with_retry"
                ):
                    wrapper_call = True
            continue
        if isinstance(node, ast.Assign):
            collect_alias(node)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "pr_create":
                raw_call = True
            elif isinstance(func, ast.Name):
                name_calls.add(func.id)
                if func.id == "create_pr_with_retry":
                    wrapper_call = True
        stack.extend(ast.iter_child_nodes(node))

    return raw_call or bool(name_calls & aliases), wrapper_call


def _imports_and_uses_validator(tree: ast.AST) -> bool:
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "closing_reference":
            if any(alias.name == "validate_closing_reference" for alias in node.names):
                imported = True
        elif isinstance(node, ast.ImportFrom) and node.module == "charlie_work.closing_reference":
            if any(alias.name == "validate_closing_reference" for alias in node.names):
                imported = True
    if not imported:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_closing_reference"
        ):
            return True
    return False


@pytest.fixture(scope="module")
def pr_create_call_sites() -> dict[Path, _FileFacts]:
    """The one per-module scan of ``_SRC_ROOT`` both AC guards share.

    Keys are paths relative to ``_SRC_ROOT``, not basenames: two same-named
    files in different packages (``__init__.py``, ``config.py``,
    ``decide.py``) can never overwrite each other and mask a call site.
    The guards below still exempt and report by basename, as before.

    rglob, not glob: a call site added under a subpackage in the future must
    not be silently exempted by scanning only the top-level directory.

    The detectors below need the literal token: a raw ``.pr_create``
    attribute call, a ``"pr_create"`` constant (the getattr shape), the
    ``def pr_create`` definition name, or a ``create_pr_with_retry`` call all
    spell their identifier in the source text. Files with neither token are
    skipped before parsed() (issue #2578; same filter shape as
    #2422/#2548).

    Both guard tests iterate the same files, run the same filter and walk
    the same trees; the second to run used to re-pay every pass against an
    already-parsed tree, and the first paid the whole-package cold
    ``_src_ast`` fill inside its own ``call`` phase (issue #3065). Running
    the scan once per module moves that cost here -- the first consumer's
    ``setup`` -- so neither test's ``call`` measures it (the #3039 remedy
    shape).
    """
    facts: dict[Path, _FileFacts] = {}
    for path in source_files(_SRC_ROOT):
        text = source_text(path)
        if "pr_create" not in text and "create_pr_with_retry" not in text:
            continue
        tree = parsed(path)
        raw_call, wrapper_call = _call_site_facts(tree)
        uses_validator = _imports_and_uses_validator(tree) if wrapper_call else False
        facts[path.relative_to(_SRC_ROOT)] = _FileFacts(raw_call, wrapper_call, uses_validator)
    return facts


def test_every_pr_create_call_site_routes_through_the_validator(
    pr_create_call_sites: dict[Path, _FileFacts],
) -> None:
    offenders: list[str] = []
    scanned_with_call_site: list[str] = []

    # cw#1273 interposed create_pr_with_retry() between every caller and the
    # raw gh.pr_create -- the raw attribute call now lives only inside
    # pr_create_retry.py (see the AC6 retry-wrapper scan below), so
    # "where does a caller request a PR creation" is now the wrapper call,
    # not the raw attribute call. Using the pre-#1273 raw-attribute detector
    # here would find zero call sites in the consumers (both call the
    # wrapper, not gh.pr_create directly) and instead flag pr_create_retry.py
    # itself -- a module that forwards an already-built body and has no
    # closing-ref concern of its own.
    for rel, facts in pr_create_call_sites.items():
        if not facts.has_wrapper_call:
            continue
        scanned_with_call_site.append(rel.name)
        if not facts.uses_validator:
            offenders.append(rel.name)

    # Positive control: this scan must actually find the two known call
    # sites (dead_worker_sweep/effects_pr.py, reconcile_salvage.py) or the
    # "zero offenders"
    # result below would be indistinguishable from "the scan never matched
    # anything". Issue #1317 moved the workflow.py call site (inside
    # _open_salvage_pr / _open_pr_for_orphaned_branch) verbatim into
    # dead_worker_reap.py, wave B (dead-worker sweep) moved it again into
    # dead_worker_sweep/effects_pr.py, and #2226 moved the reconcile lane's
    # call site into reconcile_salvage.py; the call site itself is
    # unchanged, only its module.
    assert "effects_pr.py" in scanned_with_call_site
    assert "reconcile_salvage.py" in scanned_with_call_site

    assert offenders == [], (
        f"Module(s) call create_pr_with_retry without routing through "
        f"validate_closing_reference first: {offenders}"
    )


# ---------------------------------------------------------------------------
# cw#1273 AC6: no `gh.pr_create` call site may bypass the bounded outer
# retry + duplicate-PR guard (`create_pr_with_retry`). Derived from the same
# AST scan as the validator check above, not a hardcoded file list, so a
# future third call site is caught automatically.
# ---------------------------------------------------------------------------


def test_every_pr_create_call_site_routes_through_the_retry_wrapper(
    pr_create_call_sites: dict[Path, _FileFacts],
) -> None:
    # pr_create_retry.py's own body is the one legitimate place a raw
    # `.pr_create(` call may live -- it *is* the routing target every other
    # module must call instead, the same way the definition of `pr_create`
    # itself (inside github.py) is excluded by `_call_site_facts` rather
    # than flagged as a call site of itself.
    bypass_offenders = [
        rel.name
        for rel, facts in pr_create_call_sites.items()
        if rel.name != "pr_create_retry.py" and facts.has_raw_call
    ]
    wrapper_consumers = [
        rel.name for rel, facts in pr_create_call_sites.items() if facts.has_wrapper_call
    ]

    # Positive control for the raw-call detector itself: pr_create_retry.py
    # must report has_raw_call (its `gh.pr_create(` sits at
    # pr_create_retry.py:358), or a detector that always returned False
    # would leave bypass_offenders empty and pass vacuously below.
    retry_facts = next(
        (f for rel, f in pr_create_call_sites.items() if rel.name == "pr_create_retry.py"),
        None,
    )
    assert retry_facts is not None and retry_facts.has_raw_call

    # Positive control: the scan must actually find real wrapper usage, or
    # an empty `bypass_offenders` below would be indistinguishable from
    # "the scan never matched anything". See the #1317 note above the other
    # positive control in this file -- the call site moved from workflow.py
    # to dead_worker_reap.py verbatim, then to dead_worker_sweep/effects_pr.py,
    # and (for the reconcile lane) to reconcile_salvage.py under #2226.
    assert "effects_pr.py" in wrapper_consumers
    assert "reconcile_salvage.py" in wrapper_consumers

    assert bypass_offenders == [], (
        f"Module(s) call gh.pr_create directly, bypassing the bounded outer "
        f"retry + duplicate-PR guard (cw#1273): {bypass_offenders}"
    )


# ---------------------------------------------------------------------------
# AC5: the post-create `closingIssuesReferences` probe, at both call sites.
#
# These exercise the actual `_open_salvage_pr` / `apply_fixes` code paths
# with `state_file`/`state_path` set (the earlier version of this test file
# only unit-tested the pure `validate_closing_reference` function, so the
# post-create logging block itself had zero coverage -- it could log
# unconditionally, or never, and every existing test would still pass,
# because `log_event` writes to events.db, not the `state["events"]` list
# most tests inspect). The negative case (matched -> no event) is the one
# that actually discriminates: a test that only checks the mismatch case
# passes even if the block logs unconditionally.
# ---------------------------------------------------------------------------


def _salvage_labels(config: OrchestratorConfig) -> tuple[set[str], set[str]]:
    active = {config.labels.in_progress}
    issue = {config.labels.in_progress}
    return active, issue


def test_open_salvage_pr_logs_unlinked_event_on_mismatch(tmp_path: Path) -> None:
    from _salvage_fixtures import _SalvageTestGitHub

    from charlie_work.workflow import _open_salvage_pr

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    # GitHub resolves the created PR against a DIFFERENT issue than the one
    # the orchestrator intended -- the real-world case this event exists to
    # catch (charlie-work's own body text looked correct; GitHub's own
    # closing-reference resolution disagreed anyway).
    gh = _SalvageTestGitHub(repo_root=tmp_path, closing_issue_numbers=[999])

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert pr_number == 101
    assert error is None
    assert gh.pr_view_calls == [101, 101, 101]  # re-probed across the indexing race (#1868)
    events = query_events(state_file, kind="pr_closing_ref_unlinked", issue_number=42)
    assert len(events) == 1
    assert events[0]["payload"]["pr_number"] == 101
    assert events[0]["payload"]["linked_issue_numbers"] == [999]


def test_open_salvage_pr_no_unlinked_event_when_matched(tmp_path: Path) -> None:
    """The discriminating case: GitHub's resolution DOES match. No event."""
    from _salvage_fixtures import _SalvageTestGitHub

    from charlie_work.workflow import _open_salvage_pr

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, closing_issue_numbers=[42])

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert pr_number == 101
    assert error is None
    events = query_events(state_file, kind="pr_closing_ref_unlinked", issue_number=42)
    assert events == []


def test_open_salvage_pr_no_unlinked_event_when_probe_fails(tmp_path: Path) -> None:
    """A transient ``gh pr view`` failure must not be logged as an unlinked
    reference -- it is indistinguishable from a real miss only in its return
    shape, not in what it means. Logging it anyway would make the warning
    untrustworthy (see advisor review, cw#1263)."""
    from _salvage_fixtures import _SalvageTestGitHub

    from charlie_work.workflow import _open_salvage_pr

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, pr_view_raises=True)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert pr_number == 101
    assert error is None
    events = query_events(state_file, kind="pr_closing_ref_unlinked", issue_number=42)
    assert events == []


def test_open_salvage_pr_skips_probe_under_dry_run(tmp_path: Path) -> None:
    """``pr_create`` returning ``0`` (dry-run sentinel) must not trigger a
    live ``gh pr view 0`` call."""
    from _salvage_fixtures import _SalvageTestGitHub

    from charlie_work.workflow import _open_salvage_pr

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, pr_create_return=0)

    pr_number, _error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert pr_number == 0
    assert gh.pr_view_calls == []


def test_open_salvage_pr_logs_rewritten_event_when_validator_changes_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_open_salvage_pr`` always builds a body with the correct issue number
    itself, so the rewrite branch is defensive in production -- exercise it
    by forcing ``validate_closing_reference`` to report a change, the same
    way a future defect in the body-builder would surface it."""
    from _salvage_fixtures import _SalvageTestGitHub

    from charlie_work.workflow import _open_salvage_pr

    def _fake_validate(
        body: str, issue_number: int, repo: str, gh: Any = None
    ) -> ValidationResult:
        return ValidationResult(
            body=f"Closes #{issue_number}\n\ncorrected",
            changed=True,
            findings=("closing reference rewritten",),
            target_issue_open=None,
        )

    monkeypatch.setattr(
        "charlie_work.dead_worker_sweep.effects_pr.validate_closing_reference", _fake_validate
    )

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, closing_issue_numbers=[42])

    pr_number, error, closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert pr_number == 101
    assert error is None
    assert closing_ref is not None and closing_ref.changed is True
    assert gh.prs_created[0]["body"] == "Closes #42\n\ncorrected"
    events = query_events(state_file, kind="pr_closing_ref_rewritten", issue_number=42)
    assert len(events) == 1
    assert events[0]["payload"]["source"] == "worker branch"


def test_apply_fixes_salvage_logs_unlinked_event_on_mismatch(tmp_path: Path) -> None:
    """Reconcile's ``session_unpublished_work_salvaged`` inline block routes
    through the same post-create probe as ``workflow.py``'s -- this exercises
    that side, which the earlier version of this test file never reached at
    all (every existing reconcile salvage test omits ``state_path``)."""
    from _reconcile_fixtures import (
        FakeGitHub,
        _issue,
        _setup_completed_worktree,
    )
    from _worktree_fixtures import _init_bare_remote_and_clone

    from charlie_work.reconcile import DriftItem, apply_fixes
    from charlie_work.state import empty_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 4200)

    config = OrchestratorConfig()

    class _FakeGitHubWithPrView(FakeGitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

        def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
            # Simulate GitHub resolving the PR against a different issue.
            return {"closingIssuesReferences": [{"number": 999}]}

    gh = _FakeGitHubWithPrView(
        prs=[],
        issues=[_issue(4200, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=4201,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=4200,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(empty_state()), encoding="utf-8")

    apply_fixes(gh, empty_state(), drift, config, state_path=state_path)

    events = query_events(state_path, kind="pr_closing_ref_unlinked", issue_number=4200)
    assert len(events) == 1
    assert events[0]["payload"]["pr_number"] == 4201
    assert events[0]["payload"]["linked_issue_numbers"] == [999]


def test_apply_fixes_salvage_no_unlinked_event_when_matched(tmp_path: Path) -> None:
    """Discriminating negative case for the reconcile-side probe."""
    from _reconcile_fixtures import (
        FakeGitHub,
        _issue,
        _setup_completed_worktree,
    )
    from _worktree_fixtures import _init_bare_remote_and_clone

    from charlie_work.reconcile import DriftItem, apply_fixes
    from charlie_work.state import empty_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 4300)

    config = OrchestratorConfig()

    class _FakeGitHubWithPrView(FakeGitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

        def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
            return {"closingIssuesReferences": [{"number": 4300}]}

    gh = _FakeGitHubWithPrView(
        prs=[],
        issues=[_issue(4300, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=4301,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=4300,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(empty_state()), encoding="utf-8")

    apply_fixes(gh, empty_state(), drift, config, state_path=state_path)

    events = query_events(state_path, kind="pr_closing_ref_unlinked", issue_number=4300)
    assert events == []


def test_apply_fixes_salvage_passes_corrected_body_to_pr_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (AC3): the value ``validate_closing_reference`` returns
    must be the exact body that reaches ``gh.pr_create`` on the reconcile.py
    path, not merely computed and discarded. Forces the validator to return
    a body distinct from reconcile.py's own hand-rolled one and asserts the
    *created PR's* body -- as observed by ``FakeGitHub.prs_created`` -- is
    the corrected value, not the pre-correction one. Silently dropping the
    ``salvage_body = closing_ref.body`` reassignment in reconcile.py would
    pass every other test in this suite (they only check for the presence
    of the *correct* issue number, which reconcile.py's own body-builder
    already writes in the untouched case) but fail this one."""
    from _reconcile_fixtures import (
        FakeGitHub,
        _issue,
        _setup_completed_worktree,
    )
    from _worktree_fixtures import _init_bare_remote_and_clone

    from charlie_work.reconcile import DriftItem, apply_fixes
    from charlie_work.state import empty_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 4400)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(4400, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=4401,
    )

    sentinel_body = "Closes #4400\n\nSENTINEL-BODY-FROM-VALIDATOR-4400"

    def _fake_validate(
        body: str, issue_number: int, repo: str, gh: Any = None
    ) -> ValidationResult:
        return ValidationResult(
            body=sentinel_body,
            changed=True,
            findings=("closing reference rewritten",),
            target_issue_open=None,
        )

    monkeypatch.setattr(
        "charlie_work.reconcile_salvage.validate_closing_reference", _fake_validate
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=4400,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    apply_fixes(gh, empty_state(), drift, config)

    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["body"] == sentinel_body


def test_real_pr_view_accepts_fields_keyword() -> None:
    """Guard against fake/real ``pr_view`` signature drift.

    Every fake ``GitHub`` used above accepts a ``fields`` keyword argument
    because that is what the post-create probe passes
    (``gh.pr_view(pr_number, fields=PR_CLOSING_ISSUES_FIELDS)``). If the real
    ``GitHub.pr_view`` ever dropped or renamed that parameter, the probe would
    raise a ``TypeError`` at call time -- caught by the broad
    ``except Exception`` in both call sites -- and ``pr_closing_ref_unlinked``
    would silently stop firing in production while every test here (which
    only exercises the fakes) stayed green. Assert the real signature
    directly so that drift fails loudly instead of quietly.
    """
    import inspect

    from charlie_work.github import GitHub

    sig = inspect.signature(GitHub.pr_view)
    assert "fields" in sig.parameters
    assert sig.parameters["fields"].kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
