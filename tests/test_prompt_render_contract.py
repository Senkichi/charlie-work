"""Render-contract test: every shipped prompt template must render
successfully against the *exact* key set its real production caller
supplies.

This is the structural guard for the bug class described in
``prompts.PromptTemplateError``'s docstring (issue #589): a whole-file
prompt override (or a template drifting on its own) kept referencing
``$review_summary`` after the orchestrator renamed the value it supplies to
``dispatch_note``, and nothing detected it -- ``safe_substitute`` silently
left the literal ``$review_summary`` text in twenty-one rendered review
packets before anyone noticed. ``render_prompt``'s ``strict`` mode
(``prompts.py``) now raises instead of doing that, but that only helps if
something actually calls it in every configuration templates ship in.
This file is that "something": it either drives the real writer end to end,
or -- where that is impractical -- pins the real caller's key set as a
constant tied to a cited ``file:line`` and asserts the template cannot
outgrow it unnoticed.

Deliberate-break check performed while writing this test (see the PR/report
for this change): a bogus ``$definitely_not_supplied`` placeholder was added
to ``worker.md``, ``test_worker_md_renders_via_real_writer`` was re-run and
failed with a ``PromptTemplateError`` naming exactly that placeholder, and
the template was then reverted. This proves the test is not a tautology.
"""

from __future__ import annotations

import ast
import json
import re
import string
from pathlib import Path

import pytest

from charlie_work import layout
from charlie_work.config import DispatchConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.paths import runtime_paths
from charlie_work.prompt_sections import section_variables
from charlie_work.prompts import TEMPLATE_DIR, render_prompt, resolve_template
from charlie_work.workflow import OrchestratorApp, _write_rework_prompt

# Templates that are never passed through render_prompt at all: grepping
# `orchestrator.md` and `fleet_burndown.md` across all of `src/` (not just
# workflow.py's render_prompt/._render call sites) turns up no reader other
# than each other's own cross-reference prose. They are standalone operator
# briefs meant to be read directly by a human/agent, not template-rendered --
# so they carry no $placeholders and are intentionally excluded below rather
# than silently uncovered.
_NEVER_RENDERED_TEMPLATES = {"orchestrator.md", "fleet_burndown.md"}


def _template_placeholders(template_name: str) -> set[str]:
    """Every ``$placeholder`` a template references, after expanding any
    ``$section_*`` partials it pulls in.

    A partial can itself reference placeholders (e.g. ``issue_metadata.md``
    references ``$issue_number``/``$issue_title``/``$issue_url``) that the
    *top-level* template text never mentions directly. ``render_prompt``'s own strict check
    (``prompts.py:87-96``) resolves exactly this expanded set against the
    caller's supplied values, so a contract test that only looked at the
    top-level template's own identifiers would miss a partial that drifted
    out of sync -- checking the un-expanded set only would under-report
    exactly the failure mode this test exists to catch.
    """
    text = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    sections = section_variables()
    identifiers = set(string.Template(text).get_identifiers())
    expanded = set(identifiers)
    for key in identifiers & set(sections):
        expanded |= set(string.Template(sections[key]).get_identifiers())
    return expanded


# ---------------------------------------------------------------------------
# Case A: the real writer is directly callable against tmp_path with fake
# inputs. Preferred per the task brief -- no hand-copied key list, the actual
# production code path is exercised.
# ---------------------------------------------------------------------------


def _unresolved_placeholders_in_output(rendered: str) -> set[str]:
    """Placeholder-shaped identifiers left in *rendered* output.

    Deliberately narrower than a bare ``"$" in rendered`` check: a partial's
    prose can legitimately contain a literal ``$`` that is not a
    ``string.Template`` placeholder at all -- e.g. ``mutation_check.md``'s
    shell snippet ``git show $(git merge-base ...)`` uses ``$(`` (command
    substitution), which ``get_identifiers()`` correctly ignores because it
    is not ``$identifier``/``${identifier}`` shaped. Using the same
    identifier extraction the render pipeline itself uses keeps this check
    aligned with what actually counts as an unresolved placeholder.
    """
    return set(string.Template(rendered).get_identifiers())


def _assert_no_default_state_dir_literal(rendered: str, *, template_name: str) -> None:
    """Assert *rendered* output contains no ``layout.DEFAULT_STATE_DIR`` literal.

    Companion to the unresolved-placeholder check: a template that hardcodes
    the default state-dir path (``.var/charlie-work``) instead of deriving
    every path from supplied values would silently embed it in worker/rework
    briefs. Every caller of this helper renders under a non-default
    ``runtime.state_dir`` override, so any such literal can only come from the
    template's own prose -- a correctly resolved path would point elsewhere --
    which is what keeps the assertion non-vacuous (issue #737: rendering under
    the DEFAULT state_dir would legitimately embed the literal as a correctly
    resolved path, making a naive check pass regardless of what the template
    hardcodes).

    Separator-normalized for Windows: ``str(Path)`` yields backslashes, so a
    rendered path reads ``.var\\charlie-work`` and a forward-slash-only check
    would pass vacuously on this host.
    """
    normalized = rendered.replace("\\", "/")
    assert layout.DEFAULT_STATE_DIR not in normalized, (
        f"{template_name} rendered output contains the default state-dir "
        f"literal {layout.DEFAULT_STATE_DIR!r} despite runtime.state_dir "
        f"being overridden -- the template must derive every path from "
        f"supplied values, never hardcode the default"
    )


def _fake_issue(number: int = 1) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": "Fake issue body.",
    }


def test_worker_md_renders_via_real_writer(tmp_path: Path) -> None:
    """worker.md's real caller is ``OrchestratorApp._write_worker_prompt``
    (``workflow.py::OrchestratorApp._write_worker_prompt``), used unmodified
    whenever ``config.dispatch.worker_template`` (default ``"worker.md"``) is
    selected -- see the `intake()` call site
    (``workflow.py::OrchestratorApp.intake``).

    Rendered under a non-default ``runtime.state_dir`` (issue #737) so the
    companion literal-absence assertion is non-vacuous: under the default
    state_dir a correctly resolved path would legitimately embed
    ``.var/charlie-work``, making a literal check pass regardless of what
    the template hardcodes."""
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt(_fake_issue())

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )
    _assert_no_default_state_dir_literal(rendered, template_name="worker.md")


def test_worker_claude_code_md_renders_via_real_writer(tmp_path: Path) -> None:
    """worker_claude_code.md is rendered by the *same* real writer,
    ``_write_worker_prompt`` (``workflow.py::OrchestratorApp._write_worker_prompt``),
    via ``config.dispatch.worker_template`` set to the api-worker template
    name.

    Post-Phase-2 (role-config Track B deleted per-issue adapter routing,
    and issue #1515 dropped the dead ``template=`` override parameter),
    ``_write_worker_prompt`` always renders ``config.dispatch.worker_template``
    -- the three live callers (``intake()``, and the dry-run preview and
    dispatch-loop branches inside ``_dispatch_impl``) all rely on that
    config knob. This test sets it to ``worker_claude_code.md`` (the
    api-worker default) to exercise that template's own placeholder
    correctness through the real writer.

    Rendered under a non-default ``runtime.state_dir`` (issue #737) so the
    companion literal-absence assertion is non-vacuous -- see
    ``test_worker_md_renders_via_real_writer`` for the rationale."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(worker_template="worker_claude_code.md"),
        runtime=RuntimeConfig(state_dir="custom-state"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt(_fake_issue())

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )
    _assert_no_default_state_dir_literal(rendered, template_name="worker_claude_code.md")


def test_rework_md_renders_via_real_writer_with_no_prior_decision(tmp_path: Path) -> None:
    """rework.md's real caller is the module-level ``_write_rework_prompt``
    (``workflow.py::_write_rework_prompt``), which delegates to
    ``_render_rework_prompt`` (``workflow.py::_render_rework_prompt``) for
    the literal ``values`` dict passed to ``render_prompt``. This exercises
    the no-verdict-on-disk shape (``required_changes_section`` resolves to
    ``""`` -- see ``_render_required_changes_section``
    (``workflow.py::_render_required_changes_section``)).

    Rendered under a non-default ``runtime.state_dir`` (issue #737) so the
    companion literal-absence assertion is non-vacuous -- see
    ``test_worker_md_renders_via_real_writer`` for the rationale. The
    ``state_file`` is threaded through ``runtime_paths`` the way the
    ``review.md`` test threads its paths, rather than hand-spelling the
    default ``.var/charlie-work/state.json``."""
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    state_file = runtime_paths(tmp_path, config.runtime.state_dir).state_file
    pr = {
        "number": 2,
        "title": "Fake PR title",
        "url": "https://example.test/pull/2",
        "headRefName": "agent/issue-1-fake",
    }

    prompt_path = _write_rework_prompt(state_file, pr, 1, "A dispatch note.", config)

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )
    _assert_no_default_state_dir_literal(rendered, template_name="rework.md")


def test_rework_md_renders_via_real_writer_with_required_changes(tmp_path: Path) -> None:
    """Same real writer, but with a ``request_changes`` verdict on disk so
    ``$required_changes_section`` resolves to non-empty content -- proves
    that branch's rendered text carries no stray placeholder either.

    Rendered under a non-default ``runtime.state_dir`` (issue #737) so the
    companion literal-absence assertion is non-vacuous -- see
    ``test_worker_md_renders_via_real_writer`` for the rationale."""
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    state_file = runtime_paths(tmp_path, config.runtime.state_dir).state_file
    pr_dir = state_file.parent / "prs" / "pr-3"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "needs work",
                "required_changes": ["Fix the thing."],
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": 3,
        "title": "Fake PR title",
        "url": "https://example.test/pull/3",
        "headRefName": "agent/issue-1-fake",
    }

    prompt_path = _write_rework_prompt(state_file, pr, 1, "A dispatch note.", config)

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )
    _assert_no_default_state_dir_literal(rendered, template_name="rework.md")


def test_worker_writer_rejects_flat_override_without_no_merge_contract(
    tmp_path: Path,
) -> None:
    """Issue #714: ``_write_worker_prompt`` must refuse to write a prompt
    whose rendered output is missing the no-merge contract — the exact
    failure mode a repo-local flat ``worker.md`` override creates when it
    drops the ``$section_no_merge_contract`` reference."""
    from charlie_work.prompts import MissingNoMergeContractError

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\nGo fix the issue. No safety sections here.\n"
        "$issue_number $branch_name\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override_dir)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    with pytest.raises(MissingNoMergeContractError) as exc_info:
        app._write_worker_prompt(_fake_issue())
    assert "issue #714" in str(exc_info.value)


def test_worker_writer_rejects_flat_override_without_conventional_title(
    tmp_path: Path,
) -> None:
    """Issue #715: ``_write_worker_prompt`` must refuse to write a prompt
    whose rendered output is missing the conventional-commit title instruction
    — the exact failure mode a repo-local flat ``worker.md`` override creates
    when it mandates a stale ``Fix #N: ...`` title format."""
    from charlie_work.prompts import MissingConventionalTitleError

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    # The override carries the no-merge contract (so the #714 guard passes)
    # but mandates a stale non-conventional-commit title format.
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: `Fix #$issue_number: <short title>`.\n"
        "$issue_number $branch_name\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override_dir)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    with pytest.raises(MissingConventionalTitleError) as exc_info:
        app._write_worker_prompt(_fake_issue())
    assert "issue #715" in str(exc_info.value)


def test_worker_writer_rejects_flat_override_without_execution_contract(
    tmp_path: Path,
) -> None:
    """Issue #717: ``_write_worker_prompt`` must refuse to write a prompt
    whose rendered output is missing the execution-contract escalation trigger
    — the exact failure mode a repo-local flat ``worker.md`` override creates
    when it drops the ``$section_execution_contract`` reference, leaving a
    blanket "never run the full local suite" prohibition with no carve-out for
    contract-changing diffs."""
    from charlie_work.prompts import MissingExecutionContractError

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    # The override carries the no-merge contract (so the #714 guard passes)
    # and the conventional-commit title instruction (so the #715 guard passes),
    # but drops the execution-contract carve-out.
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n"
        "Never run the full local suite as your gate -- CI runs the full "
        "matrix on push and is the sole authority on wider regressions.\n"
        "$issue_number $branch_name\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override_dir)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    with pytest.raises(MissingExecutionContractError) as exc_info:
        app._write_worker_prompt(_fake_issue())
    assert "issue #717" in str(exc_info.value)


def test_rework_writer_rejects_flat_override_without_no_merge_contract(
    tmp_path: Path,
) -> None:
    """Issue #714: ``_write_rework_prompt`` must refuse to write a rework
    brief whose rendered output is missing the no-merge contract."""
    from charlie_work.prompts import MissingNoMergeContractError

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "rework.md").write_text(
        "# Rework Task\n\nFix the PR. No safety sections here.\n"
        "$pr_number $pr_title $pr_url $issue_number $branch_name\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override_dir)))
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"
    pr = {
        "number": 2,
        "title": "Fake PR title",
        "url": "https://example.test/pull/2",
        "headRefName": "agent/issue-1-fake",
    }

    with pytest.raises(MissingNoMergeContractError) as exc_info:
        _write_rework_prompt(state_file, pr, 1, "A dispatch note.", config)
    assert "issue #714" in str(exc_info.value)


def test_rework_writer_rejects_flat_override_without_execution_contract(
    tmp_path: Path,
) -> None:
    """Issue #717: ``_write_rework_prompt`` must refuse to write a rework
    brief whose rendered output is missing the execution-contract escalation
    trigger."""
    from charlie_work.prompts import MissingExecutionContractError

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    # The override carries the no-merge contract (so the #714 guard passes)
    # but drops the execution-contract carve-out.
    (override_dir / "rework.md").write_text(
        "# Rework Task\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n"
        "Never run the full local suite as your gate -- CI runs the full "
        "matrix on push and is the sole authority on wider regressions.\n"
        "$pr_number $pr_title $pr_url $issue_number $branch_name\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override_dir)))
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"
    pr = {
        "number": 2,
        "title": "Fake PR title",
        "url": "https://example.test/pull/2",
        "headRefName": "agent/issue-1-fake",
    }

    with pytest.raises(MissingExecutionContractError) as exc_info:
        _write_rework_prompt(state_file, pr, 1, "A dispatch note.", config)
    assert "issue #717" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Case B: the real caller is a `render_prompt`/`self._render` call embedded
# deep inside a large method (`review()`) that would need extensive
# GitHub-API and subprocess mocking to invoke end to end. Per the task
# brief's documented fallback:
# pin the caller's exact key set as a constant tied to the cited file:line,
# then (a) assert the template's referenced placeholders are a subset of it
# -- so the test fails the moment a template outgrows it -- and (b) actually
# render against it, so a bug in render_prompt's own strict-check logic
# can't hide behind the subset assertion alone.
# ---------------------------------------------------------------------------

# workflow.py::OrchestratorApp.review, the literal `values` dict passed to
# `self._render("review.md", {...})`.
REVIEW_MD_SUPPLIED_KEYS = {
    "pr_number",
    "pr_title",
    "pr_url",
    "issue_number",
    "issue_title",
    "issue_url",
    "pr_json_path",
    "diff_path",
    "janitor_section",
    "test_adequacy_section",
    "static_probe_section",
    "diff_size_section",
    "ci_status_section",
    "over_cap_section",
    "attachment_budget_section",
    "prior_review_section",
}

_PINNED_KEY_SETS = {
    "review.md": REVIEW_MD_SUPPLIED_KEYS,
}


def test_pinned_templates_placeholders_are_subset_of_real_caller_keys() -> None:
    for template_name, supplied_keys in _PINNED_KEY_SETS.items():
        required = _template_placeholders(template_name)
        missing = required - supplied_keys
        assert not missing, (
            f"{template_name} references placeholder(s) {sorted(missing)} that its "
            f"real caller (see the comment above its entry in _PINNED_KEY_SETS) does "
            f"not supply -- this is the issue #589 bug class: a template drifted out "
            f"of sync with the values its production caller actually provides."
        )


def test_pinned_templates_actually_render_against_real_caller_keys() -> None:
    for template_name, supplied_keys in _PINNED_KEY_SETS.items():
        values = {key: f"<{key}>" for key in supplied_keys}
        rendered = render_prompt(template_name, values)
        assert not _unresolved_placeholders_in_output(rendered), (
            f"{template_name} rendered with an unresolved $placeholder despite strict "
            f"mode -- investigate before trusting this contract test"
        )


def test_review_md_renders_with_production_paths_and_no_state_dir_literal(
    tmp_path: Path,
) -> None:
    """review.md's real caller (``workflow.py::OrchestratorApp.review``) is
    already subset- and render-tested above against *synthetic*
    ``f"<{key}>"`` values. This test additionally builds production-shaped
    values -- real ``Path`` objects for ``pr_json_path``/``diff_path``,
    mirroring the same method's
    ``pr_dir = self.paths.prs / f"pr-{pr_number}"``
    -- under a non-default ``runtime.state_dir`` override, then asserts the
    rendered prompt contains neither an unresolved placeholder nor the
    default state-dir literal.

    This closes the half of the plan's 13a-8 requirement ("rendered output
    contains no ``.var/charlie-work`` literal") that no test in this file
    checked for *any* of the four required templates -- not something
    specific to review.md; see this change's report for the other three.

    A non-default state_dir is deliberate: rendering under the DEFAULT
    state_dir would legitimately embed ``layout.DEFAULT_STATE_DIR`` in
    ``pr_json_path``/``diff_path`` as a *correct* resolved path, making a
    "literal absent" assertion pass vacuously regardless of whether the
    template itself hardcodes anything. Overriding state_dir means every
    path threaded through the template now points elsewhere, so the
    assertion is actually exercising the template's own prose.
    """
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-42"

    values = {
        "pr_number": 42,
        "pr_title": "Fake PR title",
        "pr_url": "https://example.test/pull/42",
        "issue_number": 7,
        "issue_title": "Fake issue title",
        "issue_url": "https://example.test/issues/7",
        "pr_json_path": pr_dir / "pr.json",
        "diff_path": pr_dir / "diff.patch",
        "janitor_section": "",
        "test_adequacy_section": "",
        "static_probe_section": "",
        "diff_size_section": "",
        "ci_status_section": "",
        "over_cap_section": "",
        "attachment_budget_section": "",
        "prior_review_section": "",
    }
    assert set(values) == REVIEW_MD_SUPPLIED_KEYS

    rendered = render_prompt("review.md", values)

    assert not _unresolved_placeholders_in_output(rendered), (
        "review.md rendered with an unresolved $placeholder against production-shaped values"
    )
    # Normalize separators before the literal check: str(Path) on Windows
    # yields backslashes, so a rendered path reads ".var\charlie-work", and
    # checking only the forward-slash spelling of DEFAULT_STATE_DIR would
    # pass vacuously on this host regardless of what the template contains.
    normalized = rendered.replace("\\", "/")
    assert layout.DEFAULT_STATE_DIR not in normalized, (
        f"review.md rendered output contains the default state-dir literal "
        f"{layout.DEFAULT_STATE_DIR!r} despite runtime.state_dir being "
        f"overridden to 'custom-state' -- the template must derive every "
        f"path from supplied values, never hardcode the default"
    )


def test_review_md_repo_local_override_render_with_no_state_dir_literal(
    tmp_path: Path,
) -> None:
    """review.md can also ship as a repo-local override (issue #589's shape).

    ``OrchestratorApp.review()`` passes ``self.prompt_dirs`` to
    ``render_prompt`` inside ``workflow.py::OrchestratorApp._render``. That
    search-dir list is empty by default, but ``runtime.prompts_dir`` can
    point at a repo-local template directory, and ``resolve_template`` picks
    a repo-local ``review.md`` over the package default. This test copies
    the packaged template into a
    temporary override directory and renders from there, asserting the same
    literal-absence as the default-path test above.
    """
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-42"

    values = {
        "pr_number": 42,
        "pr_title": "Fake PR title",
        "pr_url": "https://example.test/pull/42",
        "issue_number": 7,
        "issue_title": "Fake issue title",
        "issue_url": "https://example.test/issues/7",
        "pr_json_path": pr_dir / "pr.json",
        "diff_path": pr_dir / "diff.patch",
        "janitor_section": "",
        "test_adequacy_section": "",
        "static_probe_section": "",
        "diff_size_section": "",
        "ci_status_section": "",
        "over_cap_section": "",
        "attachment_budget_section": "",
        "prior_review_section": "",
    }
    assert set(values) == REVIEW_MD_SUPPLIED_KEYS

    override_dir = tmp_path / "repo-local-prompts"
    override_dir.mkdir()
    (override_dir / "review.md").write_text(
        (TEMPLATE_DIR / "review.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert resolve_template("review.md", search_dirs=(override_dir,)) == (
        override_dir / "review.md"
    ), "repo-local review.md override was not selected by resolve_template"

    rendered = render_prompt("review.md", values, search_dirs=(override_dir,))

    assert not _unresolved_placeholders_in_output(rendered), (
        "repo-local review.md rendered with an unresolved $placeholder"
    )
    normalized = rendered.replace("\\", "/")
    assert layout.DEFAULT_STATE_DIR not in normalized, (
        f"repo-local review.md contains the default state-dir literal "
        f"{layout.DEFAULT_STATE_DIR!r} -- the repo-local override must "
        f"derive every path from supplied values, never hardcode the default"
    )


# ---------------------------------------------------------------------------
# Completeness: fail loudly if a new template ships without a corresponding
# contract case above (Case A or Case B), rather than silently letting it
# escape this test's coverage.
# ---------------------------------------------------------------------------

_COVERED_TEMPLATES = {
    "worker.md",
    "worker_claude_code.md",
    "rework.md",
    *_PINNED_KEY_SETS,
}


def test_every_shipped_template_is_covered_by_this_contract() -> None:
    shipped = {path.name for path in TEMPLATE_DIR.glob("*.md")}
    uncovered = shipped - _COVERED_TEMPLATES - _NEVER_RENDERED_TEMPLATES
    assert not uncovered, (
        f"new template(s) {sorted(uncovered)} added to prompts/ with no "
        f"render-contract coverage in this file -- add a case for its real "
        f"caller (Case A: drive the writer directly; Case B: pin the "
        f"caller's key set with a cited file:line)"
    )


# ---------------------------------------------------------------------------
# Citation drift guard (issues #1054, #1045, #1205, #1213): the
# ``workflow.py::<Symbol.path>`` references in this file's docstrings and
# comments are hand-maintained prose pointing at the real production code
# each test/constant exercises.
#
# This guard used to anchor citations to absolute line numbers
# (``workflow.py:N`` / ``workflow.py:N-M``). That anchoring drifted
# repeatedly: issue #740 / PR #1045 fixed a batch of stale line citations
# and *introduced* a new wrong one in the same diff, a later re-sync
# (commit 90bb8d8) drifted again as workflow.py grew, and #1205 documented
# yet another instance of a stale-citation fix introducing new staleness.
# Worse, because nearly every fleet PR touches workflow.py somewhere (a
# 22k-line file), any insertion *above* a cited line staled every citation
# below it -- turning this guard into a mechanical CI tax on PRs whose
# diffs never went near the cited code, rather than a genuine drift signal
# (#1213).
#
# #1213's fix: anchor citations to a **symbol** (a function/method's
# qualified name, e.g. ``OrchestratorApp._write_worker_prompt``) instead of
# a line range, resolved dynamically via an AST walk of workflow.py. A
# symbol's *position* moves around just like any line number does, but the
# citation now names WHAT is being pointed at rather than WHERE it
# currently sits, so it survives unrelated insertions anywhere else in the
# file. The marker-substring check from the original #1054 guard is
# preserved unchanged in spirit -- it is what still catches a citation
# pointing at the wrong symbol (a rename, a copy-paste into the wrong
# docstring, a symbol whose body no longer does what the prose claims).
# ---------------------------------------------------------------------------

# Each entry maps a ``workflow.py::<Symbol.path>`` reference to the marker
# substring(s) that MUST appear somewhere in that symbol's source span
# (from its `def`/`class` line through its last body line, per
# ``ast.FunctionDef``/``ast.AsyncFunctionDef``/``ast.ClassDef``
# ``.lineno``/``.end_lineno``). A symbol cited for more than one thing
# inside it (e.g. ``OrchestratorApp.review`` is cited both for its
# ``"review.md"`` render call and its ``pr_dir`` assignment) lists all of
# its required markers in one tuple. The substrings are chosen to uniquely
# identify the intended target within the symbol: a specific call
# expression, an assignment, or a template-name string literal. If you add
# or move a citation in this file, add/update the corresponding entry here
# -- the test will tell you if you forget.
_CITATION_EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "OrchestratorApp._write_worker_prompt": ("def _write_worker_prompt",),
    "OrchestratorApp.intake": ("self._write_worker_prompt(full_issue",),
    "_write_rework_prompt": ("def _write_rework_prompt",),
    "_render_rework_prompt": ("def _render_rework_prompt",),
    "_render_required_changes_section": ("def _render_required_changes_section",),
    "OrchestratorApp.review": (
        '"review.md"',
        "pr_dir = self.paths.prs",
    ),
    "OrchestratorApp._render": (
        "render_prompt(template_name, values, search_dirs=self.prompt_dirs)",
    ),
}

# Matches `workflow.py::<Symbol>` / `workflow.py::<Class>.<method>` -- a dotted
# path of Python identifiers naming the cited function/method/class.
_CITATION_RE = re.compile(r"workflow\.py::([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")

# The old absolute-line-number citation form #1213 replaced. Nothing in
# this file should match this again -- a hit means either a citation that
# missed the #1213 conversion, or someone reintroducing the line-anchored
# form the conversion removed on purpose.
_STALE_LINE_CITATION_RE = re.compile(r"workflow\.py:\d+(?:-\d+)?\b")


def _workflow_py_source() -> tuple[list[str], ast.Module]:
    """Read workflow.py's source lines and parsed AST."""
    import charlie_work.workflow as workflow_module

    src_path = Path(workflow_module.__file__)
    text = src_path.read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text)


def _rework_prompts_py_source() -> tuple[list[str], ast.Module]:
    """Read rework_prompts.py's source lines and parsed AST.

    Issue #1283 Phase A moved three of this file's cited symbols
    (``_write_rework_prompt``, ``_render_rework_prompt``,
    ``_render_required_changes_section``) out of workflow.py into this
    module. The citation strings in ``_CITATION_EXPECTATIONS`` still read
    ``workflow.py::<Symbol>`` -- the facade re-export keeps the public name
    stable -- so resolution must consider both files, not just workflow.py.
    """
    import charlie_work.rework_prompts as rework_prompts_module

    src_path = Path(rework_prompts_module.__file__)
    text = src_path.read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text)


def _resolve_symbols(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """Map every ``Class.method`` / module-level ``function`` qualified name
    defined in workflow.py's AST to its ``(lineno, end_lineno)`` span
    (1-indexed, inclusive), by walking ``ClassDef``/``FunctionDef``/
    ``AsyncFunctionDef`` nesting.

    A name defined more than once (e.g. two classes each with a same-named
    method, which would collide under this flat qualname scheme) is
    deliberately dropped from the result rather than resolved to either
    definition -- the caller reports that as "ambiguous", not as a silent
    pick of whichever definition happened to be seen first.
    """
    occurrences: dict[str, list[tuple[int, int]]] = {}

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = ".".join(stack + [child.name])
                occurrences.setdefault(qualname, []).append((child.lineno, child.end_lineno))
                walk(child, stack + [child.name])
            else:
                walk(child, stack)

    walk(tree, [])
    return {name: spans[0] for name, spans in occurrences.items() if len(spans) == 1}


def _collect_citation_failures(
    source_lines: list[str],
    expectations: dict[str, tuple[str, ...]],
    symbol_spans: dict[str, tuple[int, int]],
    wf_lines: list[str],
) -> list[str]:
    """Core citation-check logic behind ``test_workflow_py_citations_are_not_stale``,
    factored out so its failure branches can be exercised directly with
    synthetic inputs -- ``test_collect_citation_failures_flags_unresolved_symbol``
    and ``test_collect_citation_failures_flags_stale_line_citation`` below --
    without mutating this file or workflow.py to force a real failure.

    ``source_lines`` plays the role of this file's own source (scanned for
    citations and old-style line anchors); ``expectations`` plays
    ``_CITATION_EXPECTATIONS``; ``symbol_spans`` plays the workflow.py AST
    resolution from ``_resolve_symbols``; ``wf_lines`` plays workflow.py's
    source lines. The real test below calls this with the real four inputs
    and its behavior is unchanged from before this refactor.
    """
    failures: list[str] = []

    # No old-style line-anchored citation may remain -- #1213 replaced the
    # whole form, not just the stale instances that prompted it.
    for line_no, line in enumerate(source_lines, 1):
        if _STALE_LINE_CITATION_RE.search(line):
            failures.append(
                f"line {line_no}: found an old-style absolute-line-number "
                f"citation ({line.strip()!r}) -- issue #1213 replaced these "
                f"with `workflow.py::<Symbol.path>` references; convert it "
                f"instead of reintroducing a line-anchored citation"
            )

    # Collect every symbol citation that appears in this file. A citation
    # with no matching expectation is itself a failure -- it means someone
    # added a citation without registering what it should contain.
    found: set[str] = set()

    for line_no, line in enumerate(source_lines, 1):
        for match in _CITATION_RE.finditer(line):
            symbol = match.group(1)
            found.add(symbol)
            markers = expectations.get(symbol)
            if markers is None:
                failures.append(
                    f"line {line_no}: citation workflow.py::{symbol} has no "
                    f"entry in _CITATION_EXPECTATIONS -- add one with the "
                    f"marker substring(s) its span must contain"
                )
                continue
            span = symbol_spans.get(symbol)
            if span is None:
                failures.append(
                    f"line {line_no}: citation workflow.py::{symbol} does not "
                    f"resolve to exactly one function/method/class in "
                    f"workflow.py -- it was either renamed/removed, or the "
                    f"name is ambiguous (defined more than once) and needs a "
                    f"more specific qualified path"
                )
                continue
            start, end = span
            cited_text = "\n".join(wf_lines[start - 1 : end])
            for marker in markers:
                if marker not in cited_text:
                    failures.append(
                        f"line {line_no}: citation workflow.py::{symbol} "
                        f"(currently workflow.py:{start}-{end}) expected to "
                        f"contain {marker!r} but it does not appear anywhere "
                        f"in that symbol's source"
                    )

    # Also fail if a registered expectation has no corresponding citation in
    # the file -- that means a citation was removed but the expectation was
    # left behind, or the symbol name was changed without updating the
    # expectation table.
    orphaned = set(expectations) - found
    for symbol in sorted(orphaned):
        failures.append(
            f"_CITATION_EXPECTATIONS has entry for workflow.py::{symbol} but "
            f"no matching citation appears in this file -- update or remove "
            f"the expectation"
        )

    return failures


def test_workflow_py_citations_are_not_stale() -> None:
    """Every ``workflow.py::<Symbol.path>`` citation in this file must resolve
    to exactly one function/method/class in workflow.py, and the marker
    substring(s) registered for it in ``_CITATION_EXPECTATIONS`` must appear
    somewhere in that symbol's source span.

    This is the structural guard for issue #1054's recurrence, now anchored
    to symbols instead of absolute line numbers (issue #1213) -- see the
    module comment above ``_CITATION_EXPECTATIONS`` for the full #1054 /
    #1045 / #1205 / #1213 lineage. Anchoring to a symbol makes a citation
    survive unrelated workflow.py edits (this guard's whole point), but that
    is not the same as the citation being *right*: the marker check still
    catches a citation that resolves cleanly but points at the wrong code.

    Issue #1283 Phase A: three cited symbols moved to
    ``charlie_work/rework_prompts.py``. Rather than change
    ``_collect_citation_failures``'s signature (a single flat
    ``symbol_spans``/``wf_lines`` pair), workflow.py's and rework_prompts.py's
    lines are concatenated and rework_prompts.py's spans are offset by
    ``len(wf_lines)`` so a single ``(start, end)`` pair still slices the
    right lines regardless of which file actually defines the symbol -- a
    name resolved in both files is dropped as ambiguous, the same rule
    ``_resolve_symbols`` already applies within one file.
    """
    wf_lines, wf_tree = _workflow_py_source()
    rp_lines, rp_tree = _rework_prompts_py_source()
    wf_spans = _resolve_symbols(wf_tree)
    rp_spans = _resolve_symbols(rp_tree)

    offset = len(wf_lines)
    combined_lines = wf_lines + rp_lines
    symbol_spans: dict[str, tuple[int, int]] = dict(wf_spans)
    for name, (start, end) in rp_spans.items():
        if name in symbol_spans:
            del symbol_spans[name]
            continue
        symbol_spans[name] = (start + offset, end + offset)

    this_file = Path(__file__)
    this_src = this_file.read_text(encoding="utf-8").splitlines()

    failures = _collect_citation_failures(
        this_src, _CITATION_EXPECTATIONS, symbol_spans, combined_lines
    )

    assert not failures, (
        "stale or missing workflow.py symbol citations (issue #1054/#1213 "
        "recurrence guard):\n" + "\n".join(failures)
    )


def test_resolve_symbols_drops_ambiguous_duplicate_qualname() -> None:
    """``_resolve_symbols`` must drop a qualname defined more than once
    rather than silently resolving it to whichever definition it saw first,
    while still resolving an unambiguous sibling symbol correctly.

    Uses a small synthetic module (not workflow.py) so the ambiguous case
    doesn't require mutating real production code to construct.
    """
    source = (
        "def foo():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 2\n"
        "\n"
        "\n"
        "def unique_top_level():\n"
        "    return 3\n"
    )
    tree = ast.parse(source)

    spans = _resolve_symbols(tree)

    assert "foo" not in spans, (
        "a qualname defined twice (duplicate top-level `def foo`) must be "
        "dropped as ambiguous, not resolved to either definition"
    )

    assert "unique_top_level" in spans
    start, end = spans["unique_top_level"]
    source_lines = source.splitlines()
    assert source_lines[start - 1].strip() == "def unique_top_level():"
    assert source_lines[end - 1].strip() == "return 3"


def test_collect_citation_failures_flags_unresolved_symbol() -> None:
    """The unresolved/renamed-symbol failure branch of
    ``test_workflow_py_citations_are_not_stale`` (empty ``symbol_spans.get``)
    must produce a failure naming the citation, not silently pass.

    The fake citation is assembled from separate pieces at runtime (see
    ``symbol``/``citation`` below) rather than written as one contiguous
    literal in this file's source: a contiguous literal would itself be
    picked up by the real guard's own citation scan
    (``test_workflow_py_citations_are_not_stale``) when it reads this file,
    and fail because no matching entry exists in ``_CITATION_EXPECTATIONS``.
    Assembling at runtime produces the joined string only in memory, which
    is what this synthetic test needs, without leaving a matching literal in
    the scanned source text.
    """
    symbol = "Nonexistent" + ".symbol"
    citation = "workflow.py::" + symbol
    source_lines = [f"# see {citation} for details"]
    expectations = {symbol: ("some marker",)}
    symbol_spans: dict[str, tuple[int, int]] = {}  # simulates a renamed/removed symbol
    wf_lines: list[str] = []

    failures = _collect_citation_failures(source_lines, expectations, symbol_spans, wf_lines)

    assert len(failures) == 1
    assert "does not resolve to exactly one function/method/class" in failures[0]
    assert symbol in failures[0]


def test_collect_citation_failures_flags_stale_line_citation() -> None:
    """``_STALE_LINE_CITATION_RE`` must actually fire and fail the guard when
    a line-anchored old-style module-plus-line-number citation is present --
    not merely fail to fire on today's (already-converted) citations.

    The stale citation string is assembled from separate pieces at runtime
    for the same reason as the previous test: a contiguous literal here
    would trip the real guard's own stale-line-citation scan when it reads
    this file.
    """
    stale_citation = "workflow.py:" + "123"
    source_lines = [f"# old style: {stale_citation}"]

    failures = _collect_citation_failures(source_lines, {}, {}, [])

    assert len(failures) == 1
    assert "old-style absolute-line-number citation" in failures[0]
    assert stale_citation in failures[0]
