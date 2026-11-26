"""Doctor: workflow shapes whose check names cannot be read statically (issue #2316).

``doctor``'s ``check name:`` cross-check matches ``auto_merge.required_checks``
against the static ``jobs.<id>.name`` values in ``.github/workflows/``. Two
legitimate CI shapes put a required context on the PR that no static job name
spells, even though the merge gate (``checks.py``, exact match on the PR's real
check runs) is satisfied:

* a step POSTs the context itself through ``gh api repos/<repo>/check-runs``
  (e.g. lint folded into the test job, which then reports the old lint context
  so branch protection and the queue config stay unchanged);
* a job's ``name:`` is a ``${{ ... }}`` expression, so its real check-run name
  only exists at run time (a kill-switch path that renames a standalone job).

:func:`scan_dynamic_check_sources` derives which of those shapes the repo
contains, from the workflow files, so doctor can downgrade a no-static-match
finding from ``error`` to ``warning`` instead of rejecting a working setup.
Nothing here is configured by hand: a repo that stops using either shape stops
getting the downgrade.

A plain stale name (no such shape anywhere) is unaffected and stays an
``error``. This module only *classifies*; the severity decision lives in
``doctor.run_doctor`` next to the match it qualifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

# ``gh api`` flags that add request parameters or a body. ``gh api`` defaults
# to POST "if any parameters were added" and to GET otherwise (``gh api
# --help``, gh 2.88.1), so a check-runs call carrying one of these posts even
# with no explicit method.
_GH_API_PARAM_FLAG = re.compile(r"(?:^|\s)(?:-f|-F|--field|--raw-field|--input)(?:[=\s]|$)")
# An explicit method flag: ``--method POST``, ``--method=POST``, ``-X POST``,
# ``-XPOST``. Group 1 is the verb.
_GH_API_METHOD_FLAG = re.compile(r"(?:^|\s)(?:--method[=\s]+|-X\s*)([A-Za-z]+)")
_CHECK_RUNS_ENDPOINT = re.compile(r"(?:^|[/\s'\"])check-runs(?:[/?\s'\"]|$)")
# ``\`` at end of line (POSIX/bash continuation) joins the next line.
_LINE_CONTINUATION = re.compile(r"\\\r?\n")


@dataclass(frozen=True)
class DynamicCheckSources:
    """Where a repo's workflows report check names a static scan cannot see.

    Both fields hold human-readable locators (``"<file>: job `<id>`"``) so the
    doctor detail names exactly which workflow to read.
    """

    posting_steps: tuple[str, ...] = ()
    expression_job_names: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.posting_steps or self.expression_job_names)

    def describe(self) -> str:
        """One sentence naming each shape found; empty when there is none."""
        parts: list[str] = []
        if self.posting_steps:
            parts.append(
                "a step POSTs to the check-runs API in "
                + ", ".join(self.posting_steps)
                + " (the context may be posted by the workflow itself)"
            )
        if self.expression_job_names:
            parts.append(
                "a job name is a `${{ ... }}` expression in "
                + ", ".join(self.expression_job_names)
                + " (the real name only exists at run time)"
            )
        return "; ".join(parts)


def is_expression_name(name: Any) -> bool:
    """True when a job ``name:`` is (or embeds) a ``${{ ... }}`` expression."""
    return isinstance(name, str) and "${{" in name


def is_check_run_post(run_text: str) -> bool:
    """True when a step's shell text creates a check run via ``gh api``.

    A logical command (backslash continuations joined) counts when it invokes
    ``gh api`` against a ``check-runs`` endpoint AND posts: an explicit
    ``POST`` method, or -- with no explicit method -- any parameter/body flag,
    which makes ``gh api`` default to POST. An explicit non-POST method (e.g.
    ``--method GET`` to *list* check runs) is a read and does not count.

    Only the ``gh api`` shape is recognised; a workflow that posts the context
    some other way (``curl``, ``actions/github-script``) is not detected and
    keeps the ``error`` severity -- the conservative direction.
    """
    for command in _LINE_CONTINUATION.sub(" ", run_text).splitlines():
        if "gh api" not in command or not _CHECK_RUNS_ENDPOINT.search(command):
            continue
        method = _GH_API_METHOD_FLAG.search(command)
        if method is not None:
            if method.group(1).upper() == "POST":
                return True
        elif _GH_API_PARAM_FLAG.search(command):
            return True
    return False


def _iter_workflow_jobs(repo_root: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(workflow file name, job id, job mapping)`` for each parseable job.

    Same discovery as ``doctor.workflow_job_names``: ``*.y*ml`` under
    ``.github/workflows/``, unreadable or unparseable files skipped.
    """
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return
    for candidate in sorted(workflows_dir.glob("*.y*ml")):
        try:
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        jobs = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if isinstance(job, dict):
                yield candidate.name, str(job_id), job


def _job_posts_check_run(job: dict[str, Any]) -> bool:
    steps = job.get("steps")
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, dict)
        and isinstance(step.get("run"), str)
        and is_check_run_post(step["run"])
        for step in steps
    )


def scan_dynamic_check_sources(repo_root: Path) -> DynamicCheckSources:
    """Find the workflow jobs that report a check name doctor cannot read statically."""
    posting: list[str] = []
    expression: list[str] = []
    for file_name, job_id, job in _iter_workflow_jobs(repo_root):
        locator = f"{file_name}: job `{job_id}`"
        if _job_posts_check_run(job):
            posting.append(locator)
        if is_expression_name(job.get("name")):
            expression.append(locator)
    return DynamicCheckSources(
        posting_steps=tuple(posting), expression_job_names=tuple(expression)
    )
