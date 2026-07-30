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

import json
import string
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.prompt_sections import section_variables
from charlie_work.prompts import TEMPLATE_DIR, render_prompt
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
    references ``$issue_number``/``$issue_title``/``$issue_url``/
    ``$worker_model_tier``) that the *top-level* template text never
    mentions directly. ``render_prompt``'s own strict check
    (``prompts.py:92-93``) resolves exactly this expanded set against the
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


def _fake_issue(number: int = 1) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": "Fake issue body.",
    }


def test_worker_md_renders_via_real_writer(tmp_path: Path) -> None:
    """worker.md's real caller is ``OrchestratorApp._write_worker_prompt``
    (workflow.py:13777-13794), used unmodified whenever
    ``config.dispatch.worker_template`` (default ``"worker.md"``) is
    selected -- see the `intake()` call site at workflow.py:5257."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt(_fake_issue())

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )


def test_worker_claude_code_md_renders_via_real_writer(tmp_path: Path) -> None:
    """worker_claude_code.md is rendered by the *same* real writer,
    ``_write_worker_prompt``, with an explicit ``template=`` override --
    exactly what the api-worker dispatch path at workflow.py:5863-5875 does
    (``template = self.config.api_worker.worker_template``, then
    ``self._write_worker_prompt(full_issue, template=template)``)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt(
        _fake_issue(), template=config.api_worker.worker_template
    )

    rendered = prompt_path.read_text(encoding="utf-8")
    assert not _unresolved_placeholders_in_output(rendered), (
        "rendered prompt still contains an unresolved $placeholder"
    )


def test_rework_md_renders_via_real_writer_with_no_prior_decision(tmp_path: Path) -> None:
    """rework.md's real caller is the module-level ``_write_rework_prompt``
    (workflow.py:3616-3664). This exercises the no-verdict-on-disk shape
    (``required_changes_section`` resolves to ``""`` -- see
    ``_render_required_changes_section``, workflow.py:3536)."""
    config = OrchestratorConfig()
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"
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


def test_rework_md_renders_via_real_writer_with_required_changes(tmp_path: Path) -> None:
    """Same real writer, but with a ``request_changes`` verdict on disk so
    ``$required_changes_section`` resolves to non-empty content -- proves
    that branch's rendered text carries no stray placeholder either."""
    config = OrchestratorConfig()
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"
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


# ---------------------------------------------------------------------------
# Case B: the real caller is a `render_prompt`/`self._render` call embedded
# deep inside a large method (`review()`, `_cross_family_for_pr()`,
# `spec_review()`) that would need extensive GitHub-API and subprocess
# mocking to invoke end to end. Per the task brief's documented fallback:
# pin the caller's exact key set as a constant tied to the cited file:line,
# then (a) assert the template's referenced placeholders are a subset of it
# -- so the test fails the moment a template outgrows it -- and (b) actually
# render against it, so a bug in render_prompt's own strict-check logic
# can't hide behind the subset assertion alone.
# ---------------------------------------------------------------------------

# workflow.py:7202-7219, inside OrchestratorApp.review() -- the literal
# `values` dict passed to `self._render("review.md", {...})`.
REVIEW_MD_SUPPLIED_KEYS = {
    "pr_number",
    "pr_title",
    "pr_url",
    "issue_number",
    "issue_title",
    "issue_url",
    "pr_json_path",
    "diff_path",
    "cross_family_section",
    "janitor_section",
    "test_adequacy_section",
    "diff_size_section",
    "ci_status_section",
    "prior_review_section",
}

# workflow.py:10386-10396, inside OrchestratorApp._cross_family_for_pr() --
# the literal `values` dict passed to `self._render("cross_family_review.md", {...})`.
CROSS_FAMILY_REVIEW_MD_SUPPLIED_KEYS = {
    "pr_number",
    "pr_title",
    "pr_url",
    "issue_number",
    "issue_title",
    "pr_json_path",
    "diff_path",
}

# workflow.py:10313-10316, inside OrchestratorApp.spec_review() -- the
# literal `values` dict passed to
# `self._render("cross_family_spec_review.md", {...})`.
CROSS_FAMILY_SPEC_REVIEW_MD_SUPPLIED_KEYS = {
    "artifact_label",
    "artifact_text",
}

_PINNED_KEY_SETS = {
    "review.md": REVIEW_MD_SUPPLIED_KEYS,
    "cross_family_review.md": CROSS_FAMILY_REVIEW_MD_SUPPLIED_KEYS,
    "cross_family_spec_review.md": CROSS_FAMILY_SPEC_REVIEW_MD_SUPPLIED_KEYS,
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
