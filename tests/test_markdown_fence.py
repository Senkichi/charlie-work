"""Issue #883: the fence around ``$issue_body`` must be computed, not literal.

The old templates wrote ` ```md ` / ` ``` ` around ``$issue_body`` as literal
text, so a body containing its own fence closed the block early -- measured at
61 of 100 open issue bodies in this repo. Everything after the body's first
``` stopped being quoted material, and ``$section_scope_contract`` renders
immediately after the block, so body text could merge with the scope contract.

Two levels are tested, deliberately:

* the CommonMark *property* on ``fenced_block`` itself -- no interior line can
  close the block -- which is what makes the fix correct rather than merely
  wider;
* the end-to-end *symptom* through the real prompt writer, which is what makes
  it wired. An assertion that ``fenced_block(body) in rendered`` alone would
  only prove the two call the same function, so the integration tests also pin
  that the scope contract lands outside the block.

The byte-identity test is the no-regression pin: a body with no fence of its
own must render exactly as it did before #883, so any prompt diff in the fleet
is attributable to a body that genuinely needed a wider fence.

The same defect was then swept for rather than assumed absent, and found once
more in ``rework.md``'s ``$dispatch_note`` -- covered by the second half of
this file. That one has a rendered instance on disk
(``prs/pr-182/rework-prompt.md``), so it is a confirmed break, not a
hypothesised one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import DispatchConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.markdown_fence import MIN_FENCE_LENGTH, fence_for, fenced_block
from charlie_work.paths import runtime_paths
from charlie_work.prompts import TEMPLATE_DIR
from charlie_work.workflow import OrchestratorApp, _write_rework_prompt

TEMPLATES = ["worker.md", "worker_claude_code.md"]

# Rendered from worker_sections/scope_contract.md, immediately after the body
# block. If the block ends early this text is no longer distinguishable from
# the issue's own content.
SCOPE_CONTRACT_HEADING = "## Scope contract"


def _issue(body: str, number: int = 1) -> dict:
    return {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": body,
    }


def _app(tmp_path: Path, config: OrchestratorConfig | None = None) -> OrchestratorApp:
    config = config or OrchestratorConfig()
    return OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, gh=None
    )


def _render(tmp_path: Path, issue: dict, template: str) -> str:
    config = OrchestratorConfig(dispatch=DispatchConfig(worker_template=template))
    return _app(tmp_path, config)._write_worker_prompt(issue).read_text(encoding="utf-8")


# --- the property that makes the fix correct -------------------------------


def test_fence_grows_past_the_longest_backtick_run() -> None:
    assert fence_for("no backticks") == "`" * MIN_FENCE_LENGTH
    assert fence_for("inline `code` only") == "`" * MIN_FENCE_LENGTH
    assert fence_for("a ``` fence") == "````"
    assert fence_for("a ````` long fence") == "``````"


def test_no_interior_line_can_close_the_block() -> None:
    """The CommonMark rule: a block closes on the first line whose fence is at
    least as long as the opener. So no interior line may start one that long.
    """
    body = "before\n```\nsneaky\n```\n````\nwider\n````\nafter"

    block = fenced_block(body, "md")
    lines = block.splitlines()
    opener = lines[0].removesuffix("md")

    assert lines[-1] == opener
    assert not any(line.strip().startswith(opener) for line in lines[1:-1]), (
        "an interior line can terminate the block -- the fence is too narrow"
    )


def test_content_is_embedded_verbatim() -> None:
    """No stripping or normalisation -- that is what preserves byte-identity."""
    body = "  leading and trailing whitespace matters  \n\n"

    assert fenced_block(body, "md") == f"```md\n{body}\n```"


def test_empty_content_still_renders_a_minimum_fence() -> None:
    assert fenced_block("", "md") == "```md\n\n```"


# --- the symptom, end to end through the real writer -----------------------


@pytest.mark.parametrize("template", TEMPLATES)
def test_a_fenced_issue_body_does_not_break_out_of_its_block(
    tmp_path: Path, template: str
) -> None:
    body = (
        "Repro:\n\n```python\nprint(1)\n```\n\n"
        "## Acceptance criteria\n\n- the block must still be quoting me here"
    )
    rendered = _render(tmp_path, _issue(body), template)

    block = fenced_block(body, "md")
    assert block in rendered, "the body block is not wired into the prompt"
    assert "````md" in rendered, "the fence did not widen for a body containing a fence"

    tail = rendered[rendered.index(block) + len(block) :]
    assert SCOPE_CONTRACT_HEADING in tail, (
        "the scope contract rendered inside the issue-body block -- the body "
        "closed the fence early, which is exactly the #883 defect"
    )
    assert "- the block must still be quoting me here" in block


@pytest.mark.parametrize("template", TEMPLATES)
def test_a_null_body_renders_an_empty_block_rather_than_crashing(
    tmp_path: Path, template: str
) -> None:
    """``body: null`` comes back from the API for a body-less issue."""
    issue = _issue("")
    issue["body"] = None

    rendered = _render(tmp_path, issue, template)

    assert "```md\n\n```" in rendered
    assert "None" not in rendered.split(SCOPE_CONTRACT_HEADING)[0].split("## Issue body")[1]


# --- no-regression pin: a fence-free body renders exactly as it did --------


def _render_via_pre_change_template(tmp_path: Path, issue: dict, template: str) -> str:
    """Render *issue* through the real writer using the pre-#883 template.

    The pre-change template is the shipped one with ``$issue_body_block``
    replaced by the literal three-backtick fence it used to carry.
    """
    prompts_dir = tmp_path / "pre-change-prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shipped = (TEMPLATE_DIR / template).read_text(encoding="utf-8")
    assert "$issue_body_block" in shipped, (
        f"{template} no longer references $issue_body_block -- this test is comparing "
        f"the template against itself and can no longer detect a regression"
    )
    (prompts_dir / template).write_text(
        shipped.replace("$issue_body_block", "```md\n$issue_body\n```"), encoding="utf-8"
    )
    config = OrchestratorConfig(
        dispatch=DispatchConfig(worker_template=template),
        runtime=RuntimeConfig(prompts_dir=str(prompts_dir)),
    )
    return _app(tmp_path / "pre", config)._write_worker_prompt(issue).read_text(encoding="utf-8")


@pytest.mark.parametrize("template", TEMPLATES)
def test_a_body_without_a_fence_renders_byte_identical_to_pre_883(
    tmp_path: Path, template: str
) -> None:
    issue = _issue("Plain prose with `inline code` and no fenced block at all.")

    after = _render(tmp_path, issue, template)
    before = _render_via_pre_change_template(tmp_path, issue, template)

    assert after == before, (
        "a body with no fence of its own must render exactly as it did before #883"
    )


@pytest.mark.parametrize("template", TEMPLATES)
def test_a_body_with_a_fence_differs_from_pre_883_only_in_the_fence(
    tmp_path: Path, template: str
) -> None:
    """The complement of the byte-identity test: prove it discriminates.

    If this rendered identically too, the byte-identity test above would be
    passing for the wrong reason.
    """
    issue = _issue("Repro:\n\n```python\nprint(1)\n```")

    after = _render(tmp_path, issue, template)
    before = _render_via_pre_change_template(tmp_path, issue, template)

    after_lines = after.splitlines()
    before_lines = before.splitlines()
    assert len(after_lines) == len(before_lines), "widening the fence must not add or remove lines"

    # Asserted line-by-line rather than as a global ``after.replace("````",
    # "```") == before``: that form rewrites every four-backtick run anywhere
    # in the prompt, so it would pass or fail for reasons unrelated to the
    # fence as soon as any other part of the template grows one.
    changed = [(b, a) for b, a in zip(before_lines, after_lines) if b != a]
    assert changed == [("```md", "````md"), ("```", "````")], (
        "a fenced body must differ from pre-#883 in exactly its two fence "
        f"lines and nothing else, but the diff was {changed}"
    )


# --- the same defect in the rework brief's $dispatch_note ------------------
#
# Found by sweeping for the same defect class rather than assumed: rework.md
# wrapped $dispatch_note in a literal fence too, and the note is reviewer
# prose, which quotes pytest output and shell commands. Measured on the
# on-disk corpus: 16 of 289 review summaries contain a fence, and
# .var/charlie-work/prs/pr-182/rework-prompt.md is a rendered instance of the
# break -- its reviewer summary closed the wrapper early, so the template's
# own closing fence *opened* a block that swallowed "## Required behavior"
# and the push-verification steps below it.

REQUIRED_BEHAVIOR_HEADING = "## Required behavior"

# Shaped after pr-182's real summary: reviewer prose quoting failing tests.
FENCED_NOTE = (
    "Round 2 is not green. The new tests still fail:\n\n"
    "```\nFAILED tests/test_worker.py::test_worker_view_is_alive - TypeError\n```\n\n"
    "Fix the constructor call before reporting done."
)


def _pr(number: int = 2) -> dict:
    return {
        "number": number,
        "title": "Fake PR title",
        "url": f"https://example.test/pull/{number}",
        "headRefName": "agent/issue-1-fake",
    }


def _render_rework(tmp_path: Path, note: str, config: OrchestratorConfig | None = None) -> str:
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"
    path = _write_rework_prompt(state_file, _pr(), 1, note, config or OrchestratorConfig())
    return path.read_text(encoding="utf-8")


def _render_rework_via_pre_change_template(tmp_path: Path, note: str) -> str:
    """Render *note* through the real writer using the pre-#883 rework.md."""
    prompts_dir = tmp_path / "pre-change-prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shipped = (TEMPLATE_DIR / "rework.md").read_text(encoding="utf-8")
    assert "$dispatch_note_block" in shipped, (
        "rework.md no longer references $dispatch_note_block -- this test is "
        "comparing the template against itself and cannot detect a regression"
    )
    (prompts_dir / "rework.md").write_text(
        shipped.replace("$dispatch_note_block", "```md\n$dispatch_note\n```"), encoding="utf-8"
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(prompts_dir)))
    return _render_rework(tmp_path / "pre", note, config)


def test_a_fenced_dispatch_note_does_not_break_out_of_its_block(tmp_path: Path) -> None:
    rendered = _render_rework(tmp_path, FENCED_NOTE)

    block = fenced_block(FENCED_NOTE, "md")
    assert block in rendered, "the dispatch note is not wired into the rework brief"
    assert "````md" in rendered, "the fence did not widen for a note containing a fence"

    tail = rendered[rendered.index(block) + len(block) :]
    assert REQUIRED_BEHAVIOR_HEADING in tail, (
        "the required-behavior section rendered inside the orchestrator-review "
        "block -- exactly the corruption observed in pr-182's brief"
    )


def test_a_dispatch_note_without_a_fence_renders_byte_identical_to_pre_883(
    tmp_path: Path,
) -> None:
    note = "Plain review prose with `inline code` and no fenced block at all."

    after = _render_rework(tmp_path, note)
    before = _render_rework_via_pre_change_template(tmp_path, note)

    assert after == before, (
        "a note with no fence of its own must render exactly as it did before #883"
    )


def test_a_fenced_dispatch_note_differs_from_pre_883_only_in_the_fence(
    tmp_path: Path,
) -> None:
    after = _render_rework(tmp_path, FENCED_NOTE)
    before = _render_rework_via_pre_change_template(tmp_path, FENCED_NOTE)

    after_lines = after.splitlines()
    before_lines = before.splitlines()
    assert len(after_lines) == len(before_lines)

    changed = [(b, a) for b, a in zip(before_lines, after_lines) if b != a]
    assert changed == [("```md", "````md"), ("```", "````")], (
        "a fenced note must differ from pre-#883 in exactly its two fence "
        f"lines and nothing else, but the diff was {changed}"
    )
