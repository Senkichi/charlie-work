"""Reviewer launch functions and the harness dispatch table.

Moved verbatim out of ``workflow.py`` (wave D, D6a). The primitive launchers
are reached through a lazy ``_wf()`` accessor (``workflow.launch_claude_worker``
etc.) so patches on the ``workflow`` module's bindings, and conftest
``wrap_launchers``, still intercept them. The ``_launch_review_*`` functions
and ``_REVIEW_LAUNCHERS`` itself are reached on *this* module by the host
launch port's Real (issue #2235 deleted the ``workflow`` re-exports).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import ApiWorkerConfig, OrchestratorConfig
from .harnesses import REVIEWER_HARNESSES


def _wf() -> Any:
    import charlie_work.workflow as wf

    return wf


def _launch_review_claude_code(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    return _wf().launch_claude_worker(
        issue_number=pr_number,
        branch=branch,
        prompt_text=prompt_text,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        config=config,
        env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        # Force-enabled for reviewers: the structured events.jsonl is needed
        # for verdict fallback parsing (issue #540) and token/turn monitoring.
        tee_stream_json=True,
        resolved_review_effort=resolved_review_effort,
        max_turns_override=max_turns_override,
        model_override=model_override,
    )


def _launch_review_devin_shell(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    # devin_shell has no notion of review-effort/turn-cap resolution (those
    # are claude-code CLI concepts -- --effort and --max-turns flags); a
    # devin-routed reviewer runs with the CLI's own defaults for both.
    return _wf().launch_devin_session(
        pr_number,
        branch,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        config=config,
        worker_env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        worker_model=model_override or "",
    )


def _launch_review_api(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    # model_override is deliberately unused here: an api-routed reviewer
    # always runs the configured provider's pinned model (see
    # api_worker.launch_api_worker's own model_override=provider.model),
    # the same as an api-routed worker -- reviewer.model has no effect on
    # this harness.
    assert api_worker_config is not None  # only None for a non-api harness
    return _wf().launch_api_worker(
        pr_number,
        branch,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        api_worker_config=api_worker_config,
        worker_env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        resolved_review_effort=resolved_review_effort,
        max_turns_override=max_turns_override,
        config=config,
    )


# Single dispatch table keyed by ``reviewer.harness`` name -- this, not a
# per-harness if/elif chain, is what ``OrchestratorApp.dispatch_reviews``
# consumes (issue #1513). Every launcher above shares one keyword-only
# signature so the call site does not need to know which positional/keyword
# convention the underlying launch function uses (``launch_claude_worker``/
# ``launch_api_worker`` take ``prompt_text``; ``launch_devin_session`` takes
# ``prompt_path``); each returns a record with ``.error``/``.pid``/
# ``.process_start_time``, which is all the post-launch handling below reads.
# The assertion is the drift guard: it fails at import time if a harness is
# ever added to (or removed from) ``harnesses.REVIEWER_HARNESSES`` without a
# matching entry here, the same pattern ``adapters._ADAPTER_DISPATCHERS``
# uses for the worker side.
_REVIEW_LAUNCHERS: dict[str, Callable[..., Any]] = {
    "claude-code": _launch_review_claude_code,
    "devin-shell": _launch_review_devin_shell,
    "api": _launch_review_api,
}

assert set(_REVIEW_LAUNCHERS) == REVIEWER_HARNESSES, (
    "review_launch._REVIEW_LAUNCHERS must launch exactly the harnesses "
    "harnesses.REVIEWER_HARNESSES declares review-capable -- keep both in sync"
)
