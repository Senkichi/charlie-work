"""Checks capability: CI check/run inspection (Track 2, issue #1585).

Cluster C of the design doc's capability segmentation (Section 3.1):
``pr_checks``, ``check_run_annotations``, ``commit_check_runs``,
``actions_job``, ``workflow_runs_for_head``, ``check_graphql_rate_limit``.

``check_graphql_rate_limit`` is an ambiguity call (Section 3.1): it wraps a
GraphQL rate probe used by the checks path, though it is transport-adjacent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator


@runtime_checkable
class ChecksLike(Protocol):
    """Structural interface for CI check/run inspection operations."""

    def pr_checks(self, number: int) -> list[dict[str, Any]] | None: ...

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]: ...

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None: ...

    def actions_job(self, job_id: int) -> dict[str, Any] | None: ...

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None: ...

    def check_graphql_rate_limit(self, threshold: int = ...) -> tuple[bool, int, int | None]: ...


class Checks(CapabilityCollaborator):
    """CI check/run inspection capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
