"""Tests for issue #713: a startup/CI static check that catches a repo-local
flat prompt override referencing a placeholder its writer no longer supplies
*before* it crashes a live dispatch.

The bug class: job-cannon's flat ``.devin/prompts/rework.md`` override kept
``$review_summary`` after commit 5844c34 (PR #661) renamed the writer's slot
to ``$dispatch_note`` / ``$required_changes_section``. ``render_prompt``'s
strict mode raises :class:`PromptTemplateError` at dispatch time, but that
exception is not caught anywhere in ``src/``, so it propagated uncaught out
of every rework dispatch call site -- the crash stayed live-armed on a
running process until the next rework dispatch actually fired. The durable
fix is a pure static check (``check_prompt_template_drift``) that resolves
each configured template the way dispatch would, extracts its
``$placeholder`` set, and asserts it is a subset of the key set the
corresponding writer supplies. It runs at supervisor startup
(``OrchestratorApp.__init__``) and in CI.

The subset direction matters: an override legitimately uses fewer
placeholders than the writer supplies (job-cannon's ``worker.md`` uses 6 of
the writer's 8 keys), so the check fails only when the template reaches for
a placeholder the writer never provides -- never when it merely ignores one
the writer does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import ApiWorkerConfig, DispatchConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import (
    OrchestratorApp,
    PromptOverrideDriftError,
    REWORK_PROMPT_KEYS,
    WORKER_PROMPT_KEYS,
    _write_rework_prompt,
    check_prompt_template_drift,
)


def _fake_issue(number: int = 1) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fake issue title",
        "url": f"https://example.test/issues/{number}",
        "body": "Fake issue body.",
    }


def _fake_pr(number: int = 2) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fake PR title",
        "url": f"https://example.test/pull/{number}",
        "headRefName": "agent/issue-1-fake",
    }


def _write_override(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# check_prompt_template_drift: the pure static check itself
# ---------------------------------------------------------------------------


def test_default_configured_templates_pass_drift_check() -> None:
    """The packaged worker/rework templates (and the api-worker defaults) must
    not trip the drift check -- they are the templates the orchestrator ships,
    so a failure here means the package itself drifted, not a repo override."""
    assert check_prompt_template_drift(OrchestratorConfig()) == []


def test_drift_check_catches_rework_override_with_stale_placeholder(
    tmp_path: Path,
) -> None:
    """The exact #713 shape: a flat ``rework.md`` override still referencing
    ``$review_summary`` after the writer renamed it to ``$dispatch_note`` /
    ``$required_changes_section``. The check must report ``review_summary``
    as unsupplied -- this is the crash that would have fired on the next
    job-cannon rework dispatch."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "rework.md",
        "# Rework PR #$pr_number\n\n$review_summary\n$pr_title $pr_url "
        "$issue_number $branch_name\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))

    errors = check_prompt_template_drift(config, search_dirs=(override,))

    rework_errors = [e for e in errors if "rework.md" in str(e) and "review_summary" in str(e)]
    assert rework_errors, (
        f"drift check did not flag the stale $review_summary in the rework override; "
        f"got errors: {errors}"
    )
    assert "review_summary" in rework_errors[0].missing


def test_drift_check_catches_worker_override_with_stale_placeholder(
    tmp_path: Path,
) -> None:
    """Same bug class on the worker side: a flat ``worker.md`` override
    referencing a placeholder the worker writer never supplies."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "worker.md",
        "# Work on #$issue_number\n\n$definitely_not_supplied $branch_name\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))

    errors = check_prompt_template_drift(config, search_dirs=(override,))

    worker_errors = [
        e for e in errors if "worker.md" in str(e) and "definitely_not_supplied" in str(e)
    ]
    assert worker_errors, (
        f"drift check did not flag the stale worker placeholder; got errors: {errors}"
    )
    assert "definitely_not_supplied" in worker_errors[0].missing


def test_drift_check_passes_for_valid_subset_override(tmp_path: Path) -> None:
    """An override that uses a *subset* of the writer's keys is legitimate
    (job-cannon's ``worker.md`` uses 6 of 8) and must not be flagged. This is
    the subset-not-equality direction the issue specifies: equality would
    produce constant false failures and get the check disabled."""
    override = tmp_path / "prompts"
    # Uses only 3 of the 8 worker keys -- a valid subset.
    _write_override(
        override,
        "worker.md",
        "# Work on #$issue_number\n\nBranch: $branch_name\nTier: $worker_model_tier\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))

    errors = check_prompt_template_drift(config, search_dirs=(override,))

    worker_errors = [e for e in errors if "worker.md" in str(e)]
    assert worker_errors == [], f"a subset override must not be flagged; got: {worker_errors}"


def test_drift_check_reports_every_unsupplied_placeholder(tmp_path: Path) -> None:
    """Reporting one stale placeholder at a time turns a single fix into a
    guess-and-retry loop -- mirrors ``test_every_orphaned_placeholder_is_reported``
    in ``test_fix_prompt_template_drift.py`` for the strict-mode path."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "rework.md",
        "$pr_number $review_summary $another_orphan $branch_name\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))

    errors = check_prompt_template_drift(config, search_dirs=(override,))

    rework_error = next(e for e in errors if "rework.md" in str(e))
    assert set(rework_error.missing) == {"another_orphan", "review_summary"}


def test_drift_check_skips_missing_template_file(tmp_path: Path) -> None:
    """A configured template name that resolves to no file is a separate
    config error (dispatch surfaces it as FileNotFoundError), not placeholder
    drift. The check skips it rather than crashing or conflating the two."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(worker_template="does-not-exist.md"),
    )

    assert check_prompt_template_drift(config) == []


def test_drift_check_covers_api_worker_template(tmp_path: Path) -> None:
    """A custom ``api_worker.worker_template`` pointing at an override with a
    stale placeholder is caught too -- the api-worker dispatch path renders
    it through the same worker writer, so it is the same crash if it drifts."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "custom_worker.md",
        "# API work #$issue_number\n\n$stale_api_placeholder\n",
    )
    config = OrchestratorConfig(
        api_worker=ApiWorkerConfig(worker_template="custom_worker.md"),
        runtime=RuntimeConfig(prompts_dir=str(override)),
    )

    errors = check_prompt_template_drift(config, search_dirs=(override,))

    assert any(
        "custom_worker.md" in str(e) and "stale_api_placeholder" in str(e) for e in errors
    ), f"api_worker.worker_template drift not flagged; got: {errors}"


# ---------------------------------------------------------------------------
# Registry-drift guard: the frozensets in src/ must match the keys the real
# writers actually pass to render_prompt. Without this, a writer could add or
# rename a key and leave the registry stale -- re-arming the very crash this
# check exists to catch. Monkeypatch render_prompt to capture the values dict
# keys, invoke the real writers, and assert an exact match.
# ---------------------------------------------------------------------------


def test_worker_prompt_keys_match_real_writer(tmp_path: Path, monkeypatch) -> None:
    """``WORKER_PROMPT_KEYS`` must be exactly the keys
    ``_write_worker_prompt`` passes to ``render_prompt`` for both shipped
    worker templates (``worker.md`` and ``worker_claude_code.md``)."""
    import charlie_work.workflow as workflow_mod

    real_render = workflow_mod.render_prompt
    captured: dict[str, set[str]] = {}

    def capturing_render(template_name, values, **kwargs):
        captured.setdefault(template_name, set()).update(values.keys())
        return real_render(template_name, values, **kwargs)

    monkeypatch.setattr(workflow_mod, "render_prompt", capturing_render)

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    app._write_worker_prompt(_fake_issue())
    app._write_worker_prompt(_fake_issue(), template=config.api_worker.worker_template)

    assert set(captured) == {"worker.md", config.api_worker.worker_template}, (
        f"expected both worker templates to be rendered; captured: {sorted(captured)}"
    )
    for template_name, keys in captured.items():
        assert keys == WORKER_PROMPT_KEYS, (
            f"{template_name}: writer supplied keys {sorted(keys)} do not exactly "
            f"match WORKER_PROMPT_KEYS {sorted(WORKER_PROMPT_KEYS)} -- the registry "
            f"has drifted from the real writer (issue #713)."
        )


def test_rework_prompt_keys_match_real_writer(tmp_path: Path, monkeypatch) -> None:
    """``REWORK_PROMPT_KEYS`` must be exactly the keys
    ``_render_rework_prompt`` passes to ``render_prompt``."""
    import charlie_work.workflow as workflow_mod

    real_render = workflow_mod.render_prompt
    captured: dict[str, set[str]] = {}

    def capturing_render(template_name, values, **kwargs):
        captured.setdefault(template_name, set()).update(values.keys())
        return real_render(template_name, values, **kwargs)

    monkeypatch.setattr(workflow_mod, "render_prompt", capturing_render)

    config = OrchestratorConfig()
    state_file = tmp_path / ".var" / "charlie-work" / "state.json"

    _write_rework_prompt(state_file, _fake_pr(), 1, "A dispatch note.", config)

    assert set(captured) == {"rework.md"}, (
        f"expected only rework.md to be rendered; captured: {sorted(captured)}"
    )
    assert captured["rework.md"] == REWORK_PROMPT_KEYS, (
        f"writer supplied keys {sorted(captured['rework.md'])} do not exactly "
        f"match REWORK_PROMPT_KEYS {sorted(REWORK_PROMPT_KEYS)} -- the registry "
        f"has drifted from the real writer (issue #713)."
    )


# ---------------------------------------------------------------------------
# Startup hook: OrchestratorApp.__init__ must fail fast on drift
# ---------------------------------------------------------------------------


def test_orchestrator_app_init_raises_on_drift_override(tmp_path: Path) -> None:
    """The startup hook must refuse to construct an ``OrchestratorApp`` whose
    configured override references a stale placeholder -- so the crash is
    caught before any dispatch, not armed on a live process."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "rework.md",
        "# Rework PR #$pr_number\n\n$review_summary\n$pr_title $pr_url "
        "$issue_number $branch_name\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    with pytest.raises(PromptOverrideDriftError) as exc_info:
        OrchestratorApp(tmp_path, paths, config, gh=None)

    assert any("review_summary" in str(e) for e in exc_info.value.errors), (
        f"PromptOverrideDriftError did not name the stale placeholder; "
        f"errors: {exc_info.value.errors}"
    )
    assert "issue #713" in str(exc_info.value)


def test_orchestrator_app_init_passes_for_default_config(tmp_path: Path) -> None:
    """The startup hook must not false-fire on the packaged templates -- a
    default config constructs cleanly, or every test and supervisor start
    that uses defaults would break."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    assert app.prompt_dirs == ()


def test_orchestrator_app_init_passes_for_valid_subset_override(tmp_path: Path) -> None:
    """A valid subset override must construct cleanly -- the startup hook
    fails only on real drift, not on legitimate overrides that use fewer
    placeholders than the writer supplies."""
    override = tmp_path / "prompts"
    _write_override(
        override,
        "worker.md",
        "# Work on #$issue_number\n\nBranch: $branch_name\n",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir=str(override)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    OrchestratorApp(tmp_path, paths, config, gh=None)
