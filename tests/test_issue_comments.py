"""Issue #872: worker prompts must carry the issue's comments.

One test per acceptance criterion, plus the two hazards the criteria imply but
do not name (fence escaping, and the frozen-config tuple invariant).

The byte-identity test is the load-bearing one and is deliberately not written
as "assert the string does not contain 'Issue comments'" -- a structural check
like that passes while a stray newline ships. It reconstructs the *pre-change*
template (the shipped one minus the placeholder token), renders both through
the real writer, and compares the full strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import DispatchConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.github import defang_closing_keywords
from charlie_work.issue_comments import _fence_for, render_issue_comments
from charlie_work.paths import runtime_paths
from charlie_work.prompts import TEMPLATE_DIR
from charlie_work.workflow import OrchestratorApp


def _issue(number: int = 1, comments: list[dict] | None = None) -> dict:
    issue: dict = {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": "Fake issue body.",
    }
    if comments is not None:
        issue["comments"] = comments
    return issue


def _comment(
    login: str = "Senkichi",
    body: str = "A correction.",
    association: str = "OWNER",
    **extra: object,
) -> dict:
    return {
        "author": {"login": login},
        "authorAssociation": association,
        "createdAt": "2026-07-30T12:00:00Z",
        "body": body,
        **extra,
    }


def _app(tmp_path: Path, config: OrchestratorConfig | None = None) -> OrchestratorApp:
    config = config or OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh=None)


# --- criterion 1: comments appear, attributed ------------------------------


def test_comments_reach_the_worker_prompt_with_attribution(tmp_path: Path) -> None:
    issue = _issue(comments=[_comment(body="Use approach B, not approach A.")])

    rendered = _app(tmp_path)._write_worker_prompt(issue).read_text(encoding="utf-8")

    assert "## Issue comments" in rendered
    assert "Use approach B, not approach A." in rendered
    assert "@Senkichi" in rendered, "comment must be attributed to its author"
    assert "OWNER" in rendered


# --- criterion 2: no comments => byte-identical ----------------------------


def _render_via_pre_change_template(tmp_path: Path, issue: dict, template: str) -> str:
    """Render *issue* through the real writer using the pre-#872 template.

    The pre-change template is the shipped one with the ``$issue_comments``
    token removed -- i.e. exactly what was on disk before this change.
    """
    prompts_dir = tmp_path / "pre-change-prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shipped = (TEMPLATE_DIR / template).read_text(encoding="utf-8")
    assert "$issue_comments" in shipped, (
        f"{template} no longer references $issue_comments -- this test is comparing "
        f"the template against itself and can no longer detect a regression"
    )
    (prompts_dir / template).write_text(shipped.replace("$issue_comments", ""), encoding="utf-8")
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(prompts_dir)))
    return (
        _app(tmp_path / "pre", config)
        ._write_worker_prompt(issue, template=template)
        .read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("template", ["worker.md", "worker_claude_code.md"])
def test_issue_without_comments_renders_byte_identical_prompt(
    tmp_path: Path, template: str
) -> None:
    issue = _issue()

    after = (
        _app(tmp_path)._write_worker_prompt(issue, template=template).read_text(encoding="utf-8")
    )
    before = _render_via_pre_change_template(tmp_path, issue, template)

    assert after == before, "an issue with no comments must render exactly as it did pre-#872"


@pytest.mark.parametrize("template", ["worker.md", "worker_claude_code.md"])
def test_issue_whose_comments_are_all_filtered_renders_byte_identical_prompt(
    tmp_path: Path, template: str
) -> None:
    """Distinct path from "no comments": the list is non-empty but nothing survives."""
    issue = _issue(comments=[_comment(login="aviator-app", association="NONE")])

    after = (
        _app(tmp_path)._write_worker_prompt(issue, template=template).read_text(encoding="utf-8")
    )
    before = _render_via_pre_change_template(tmp_path, issue, template)

    assert after == before


# --- criterion 3: bounded, and truncation is announced ---------------------


def test_truncation_is_announced_and_keeps_the_newest() -> None:
    comments = [_comment(body=f"comment {i}") for i in range(10)]

    rendered = render_issue_comments(comments, max_comments=3, max_chars=0)

    assert "comment 9" in rendered, "the newest comment must survive truncation"
    assert "comment 0" not in rendered
    assert "7 earlier comment(s) omitted" in rendered, "truncation must be visible"


def test_char_budget_also_announces_and_keeps_the_newest() -> None:
    comments = [_comment(body="x" * 400) for _ in range(10)]

    rendered = render_issue_comments(comments, max_comments=0, max_chars=1000)

    assert "omitted to fit the prompt budget" in rendered
    assert rendered.count("### @Senkichi") < 10


def test_a_single_oversized_comment_is_kept_rather_than_dropped() -> None:
    """An empty section would be a worse failure than an oversized one."""
    rendered = render_issue_comments([_comment(body="y" * 5000)], max_chars=100)

    assert "yyyy" in rendered


# --- criterion 4: sanitised on the way in ----------------------------------


def test_closing_keywords_are_defanged() -> None:
    comments = [_comment(body="This closes #4321 once merged.")]

    rendered = render_issue_comments(comments, sanitize=defang_closing_keywords)

    assert "closes #4321" not in rendered.lower(), (
        "an un-defanged closing keyword can auto-close an unrelated issue when a "
        "worker copies the comment into its PR body"
    )
    assert "4321" in rendered, "defanging must preserve the reference, not delete it"


# --- criterion 5: precedence is stated -------------------------------------


def test_prompt_states_comments_take_precedence_over_the_body(tmp_path: Path) -> None:
    issue = _issue(comments=[_comment()])

    rendered = _app(tmp_path)._write_worker_prompt(issue).read_text(encoding="utf-8")

    assert "the comment wins" in rendered
    assert "latest comment wins" in rendered


# --- criterion 6: explicit author filter -----------------------------------


def test_bots_are_excluded_by_association_not_by_assuming_none_exist() -> None:
    comments = [
        _comment(login="Senkichi", body="human correction", association="OWNER"),
        _comment(login="aviator-app", body="merge queued", association="NONE"),
        _comment(login="drive-by", body="drive-by opinion", association="CONTRIBUTOR"),
    ]

    rendered = render_issue_comments(comments)

    assert "human correction" in rendered
    assert "merge queued" not in rendered
    assert "drive-by opinion" not in rendered


def test_excluded_authors_are_dropped_even_when_association_allows_them() -> None:
    comments = [
        _comment(login="charlie-bot", body="orchestrator chatter", association="COLLABORATOR"),
        _comment(login="Senkichi", body="human correction"),
    ]

    rendered = render_issue_comments(comments, excluded_authors=("CHARLIE-BOT",))

    assert "orchestrator chatter" not in rendered, "exclusion must be case-insensitive"
    assert "human correction" in rendered


def test_minimized_comments_are_dropped() -> None:
    comments = [_comment(body="hidden by a human", isMinimized=True)]

    assert render_issue_comments(comments) == ""


def test_viewer_authored_comments_are_kept() -> None:
    """The orchestrator authenticates as the operator's own account, so
    ``viewerDidAuthor`` is true for exactly the human corrections this feature
    exists to deliver. Using it as a self-filter would invert the intent -- this
    test pins that it is not used that way.
    """
    comments = [_comment(body="operator correction", viewerDidAuthor=True)]

    assert "operator correction" in render_issue_comments(comments)


# --- hazards the criteria imply --------------------------------------------


def test_nested_code_fence_cannot_break_out_of_the_comment_block() -> None:
    body = "Try this:\n\n```python\nprint(1)\n```\n\nThat's all."

    rendered = render_issue_comments([_comment(body=body)])

    fence_line = next(line for line in rendered.splitlines() if line.endswith("md"))
    fence = fence_line[:-2]
    assert len(fence) > 3, "fence must be wider than the ``` inside the comment"
    assert rendered.rstrip().endswith(fence)


def test_fence_width_grows_with_the_longest_backtick_run() -> None:
    assert _fence_for("no backticks") == "```"
    assert _fence_for("inline `code`") == "```"
    assert _fence_for("a ``` fence") == "````"
    assert _fence_for("a ````` long fence") == "``````"


def test_comment_config_sequences_are_coerced_to_tuples() -> None:
    """The frozen-dataclass invariant: a list would make the instance unhashable."""
    config = DispatchConfig(
        worker_prompt_comment_associations=["OWNER"],
        worker_prompt_excluded_comment_authors="solo-bot",
    )

    assert config.worker_prompt_comment_associations == ("OWNER",)
    # A bare string must be wrapped, not iterated into single characters.
    assert config.worker_prompt_excluded_comment_authors == ("solo-bot",)
    assert hash(config) is not None


def test_malformed_comments_are_skipped_not_crashed_on() -> None:
    comments = [
        {"authorAssociation": "OWNER", "body": "no author key"},
        {"author": None, "authorAssociation": "OWNER", "body": "null author"},
        {"author": {"login": "Senkichi"}, "authorAssociation": "OWNER", "body": "   "},
        _comment(body="the good one"),
    ]

    rendered = render_issue_comments(comments)

    assert "the good one" in rendered
    assert "@unknown" in rendered, "a comment with no author is kept but marked unknown"


def test_placeholders_inside_a_comment_are_not_expanded(tmp_path: Path) -> None:
    """Comments are a second attacker-controlled value reaching the prompt.

    ``render_prompt`` resolves partials first and then substitutes once, so a
    ``$section_*`` token inside a comment must survive as literal text rather
    than being expanded in a second pass -- the issue #8 guarantee, extended to
    the channel this change opens.
    """
    issue = _issue(comments=[_comment(body="Try $section_scope_contract and $issue_number")])

    rendered = _app(tmp_path)._write_worker_prompt(issue).read_text(encoding="utf-8")

    assert "Try $section_scope_contract and $issue_number" in rendered
