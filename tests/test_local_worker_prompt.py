"""Tests for the local-file issue source's worker prompt: the
``dispatch.worker_template`` re-default in ``load_config``, the #713
placeholder-drift check against that re-defaulted config, and
``worker_local.md``'s own rendered-output contract (the four post-render
guards, no unresolved placeholders, no push/PR instructions, and the
worker-declared-blocked-outcome section every template ships).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import (
    LOCAL_WORKER_TEMPLATE,
    DispatchConfig,
    LocalIssuesConfig,
    OrchestratorConfig,
    RuntimeConfig,
    load_config,
)
from charlie_work.paths import runtime_paths
from charlie_work.prompts import (
    NO_MERGE_CONTRACT_VARIANTS,
    MissingNoMergeContractError,
    assert_conventional_commit_title,
    assert_containment,
    assert_execution_contract,
    assert_no_merge_contract,
)
from charlie_work.workflow import OrchestratorApp, check_prompt_template_drift

from _prompt_sections_fixtures import unresolved_rendered_identifiers

# ---------------------------------------------------------------------------
# Inlined helper -- self-contained per this repo's test-file convention.
# ---------------------------------------------------------------------------


def _fake_issue(number: int = 1) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": "Fake issue body.",
    }


def _unresolved_placeholders(rendered: str) -> set[str]:
    # Issue #1780: the scratch-dir section intentionally emits a literal
    # ``$TMPDIR`` (``$$TMPDIR`` in source); the shared test helper exempts
    # exactly the identifiers the ``$$IDENT`` escapes declare.
    return unresolved_rendered_identifiers(rendered)


# ---------------------------------------------------------------------------
# load_config re-default: dispatch.worker_template -> worker_local.md
# ---------------------------------------------------------------------------


def test_load_config_local_issues_redefaults_worker_template(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.dispatch.worker_template == LOCAL_WORKER_TEMPLATE


def test_load_config_explicit_worker_template_wins_over_local_redefault(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\ndispatch:\n  worker_template: worker.md\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.dispatch.worker_template == "worker.md"


def test_load_config_local_issues_disabled_keeps_default_worker_template(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: false\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.dispatch.worker_template == "worker.md"


def test_load_config_no_local_issues_section_keeps_default_worker_template() -> None:
    config = load_config(None)

    assert config.local_issues.enabled is False
    assert config.dispatch.worker_template == "worker.md"


def test_load_config_redefault_is_load_config_specific_not_dataclass_default() -> None:
    """MUTATION CHECK (contrast) for
    ``test_load_config_local_issues_redefaults_worker_template``: building the
    identical ``local_issues.enabled=True`` state directly via the dataclass
    constructors -- bypassing ``load_config``'s data-building re-default
    branch (``config.py``'s
    ``if local_issues.enabled and "worker_template" not in dispatch_data``)
    entirely -- must NOT re-default. This proves the ``load_config`` test
    actually depends on that specific branch rather than on some inherent
    property of ``LocalIssuesConfig``/``DispatchConfig`` that would make it
    pass for an unrelated reason: if the branch were deleted, ``load_config``'s
    result would collapse to this direct-construction result (``"worker.md"``),
    which is exactly the value that would fail the real re-default test's
    ``== LOCAL_WORKER_TEMPLATE`` assertion."""
    direct = OrchestratorConfig(local_issues=LocalIssuesConfig(enabled=True))

    assert direct.dispatch.worker_template == "worker.md"


def test_check_prompt_template_drift_clean_for_local_config(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")
    config = load_config(config_file)
    assert config.dispatch.worker_template == LOCAL_WORKER_TEMPLATE

    errors = check_prompt_template_drift(config)

    assert errors == []


# ---------------------------------------------------------------------------
# worker_local.md: rendered-output contract, via the real writer
# ---------------------------------------------------------------------------


def test_worker_local_md_renders_via_real_writer_with_all_guards(tmp_path: Path) -> None:
    """worker_local.md's real caller is the same writer as worker.md --
    ``OrchestratorApp._write_worker_prompt``
    (``orchestration/prompt_ops.py::_write_worker_prompt``) -- selected via
    ``config.dispatch.worker_template``. Driving the real writer (rather than
    hand-listing ``WORKER_PROMPT_KEYS``) supplies exactly the key set the
    writer itself builds, so this can't silently drift out of sync with it."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(worker_template="worker_local.md"),
        runtime=RuntimeConfig(state_dir="custom-state"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt(_fake_issue())
    rendered = prompt_path.read_text(encoding="utf-8")

    # _write_worker_prompt already enforces these at the dispatch boundary
    # (issues #714/#715/#717/#1010); re-asserting here pins them as
    # behavioural properties of the rendered TEXT for this specific
    # template, not merely "the writer didn't raise".
    assert_no_merge_contract(rendered)
    assert_conventional_commit_title(rendered)
    assert_execution_contract(rendered)
    assert_containment(rendered)

    unresolved = _unresolved_placeholders(rendered)
    assert not unresolved, f"unresolved $placeholder(s) left in rendered output: {unresolved}"
    assert "$section_" not in rendered

    assert "git push origin" not in rendered
    assert "gh pr create" not in rendered
    assert "Worker-declared blocked outcome" in rendered


def test_local_config_end_to_end_prompt_never_orders_a_push(tmp_path: Path) -> None:
    """config file -> ``load_config`` -> real writer -> rendered text.

    The tests above each pin one link: the re-default compares against
    ``LOCAL_WORKER_TEMPLATE`` (so it follows that constant wherever it points),
    and the render test names the template literally (so it never consults the
    config). Neither fails if the constant is re-pointed at ``worker.md`` --
    found by exactly that mutation. This asserts the property the feature
    exists for, through the whole chain and without naming a template: a
    worker dispatched in a repo with no remote is never told to push.

    Positive control: the same chain with ``local_issues`` disabled DOES carry
    the push order, so the absence below is evidence, not a vacuous grep.
    """

    def rendered_for(config_text: str, root: Path) -> str:
        root.mkdir()
        config_file = root / "orchestrator.config.yaml"
        config_file.write_text(config_text, encoding="utf-8")
        config = load_config(config_file)
        app = OrchestratorApp(root, runtime_paths(root, config.runtime.state_dir), config, gh=None)
        return app._write_worker_prompt(_fake_issue()).read_text(encoding="utf-8")

    github_prompt = rendered_for("local_issues:\n  enabled: false\n", tmp_path / "github")
    local_prompt = rendered_for("local_issues:\n  enabled: true\n", tmp_path / "local")

    assert "git push origin" in github_prompt  # control
    assert "git push origin" not in local_prompt
    assert "gh pr create" not in local_prompt
    assert "Your deliverable ENDS at committing to your branch" in local_prompt


# ---------------------------------------------------------------------------
# assert_no_merge_contract: variant coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", NO_MERGE_CONTRACT_VARIANTS)
def test_assert_no_merge_contract_passes_for_each_full_variant(
    variant: tuple[str, ...],
) -> None:
    prompt = "\n\n".join(variant)
    assert_no_merge_contract(prompt)  # must not raise


def test_assert_no_merge_contract_raises_for_heading_only() -> None:
    prompt = "## No-merge contract\n\nSome unrelated prose that names neither variant.\n"
    with pytest.raises(MissingNoMergeContractError):
        assert_no_merge_contract(prompt)


def test_assert_no_merge_contract_raises_for_empty_prompt() -> None:
    with pytest.raises(MissingNoMergeContractError):
        assert_no_merge_contract("")


# ---------------------------------------------------------------------------
# Regression: worker.md / worker_claude_code.md still carry the shared
# blocked-outcome section after the local_no_merge_contract split.
# ---------------------------------------------------------------------------


def test_worker_and_api_worker_templates_still_render_blocked_outcome_section(
    tmp_path: Path,
) -> None:
    for template_name in ("worker.md", "worker_claude_code.md"):
        repo_root = tmp_path / template_name
        config = OrchestratorConfig(
            dispatch=DispatchConfig(worker_template=template_name),
            runtime=RuntimeConfig(state_dir="custom-state"),
        )
        paths = runtime_paths(repo_root, config.runtime.state_dir)
        app = OrchestratorApp(repo_root, paths, config, gh=None)

        prompt_path = app._write_worker_prompt(_fake_issue())
        rendered = prompt_path.read_text(encoding="utf-8")

        assert "Worker-declared blocked outcome" in rendered, template_name
        assert ".worker-outcome.json" in rendered, template_name
        assert "$section_" not in rendered, template_name
