"""Capability collaborators for the ``GitHub`` facade.

Track 2 of the god-object paydown campaign (issue #1585; design doc
``docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md``).
Each module holds one capability's collaborator class plus (except
``transport``) the ``@runtime_checkable`` sub-protocol declaring that
capability's slice of ``GitHubLike``. ``charlie_work.github`` builds its
``_ROUTES`` delegation table and the redeclared ``GitHubLike`` union from
these exports.

L01 (this leaf) is pure infrastructure: every collaborator class below is
still empty, and every sub-protocol's signatures are copied verbatim from
the (still monolithic) ``GitHub`` implementation. No method body has moved.
"""

from __future__ import annotations

from ._base import (
    GitHubRunResult,
    RUN_LIST_FIELDS,
    _is_mutating,
    _LIST_LIMIT,
)
from .checks import PR_CHECKS_FIELDS, Checks, ChecksLike, _job_id_from_link
from .comments import Comments, CommentsLike
from .issues import (
    ISSUE_LIST_FIELDS,
    ISSUE_VIEW_FIELDS,
    Issues,
    IssuesLike,
    get_github_issue_dependencies,
)
from .labels import LABEL_LIST_FIELDS, Labels, LabelsLike
from .merge_branch import _ADMIN_FLAG, _STRATEGY_FLAGS, MergeBranch, MergeBranchLike
from .pull_requests import (
    MERGED_PR_LIST_FIELDS,
    MergedPRSearchResult,
    PR_LIST_FIELDS,
    PR_VIEW_FIELDS,
    PullRequests,
    PullRequestsLike,
    _pr_number_from_url,
)
from .repo_meta import RepoMeta, RepoMetaLike
from .transport import RECONCILE_ISSUE_FIELDS, RECONCILE_PR_FIELDS, Transport

__all__ = [
    "Checks",
    "ChecksLike",
    "Comments",
    "CommentsLike",
    "GitHubRunResult",
    "ISSUE_LIST_FIELDS",
    "ISSUE_VIEW_FIELDS",
    "Issues",
    "IssuesLike",
    "LABEL_LIST_FIELDS",
    "Labels",
    "LabelsLike",
    "MERGED_PR_LIST_FIELDS",
    "MergeBranch",
    "MergeBranchLike",
    "MergedPRSearchResult",
    "PR_CHECKS_FIELDS",
    "PR_LIST_FIELDS",
    "PR_VIEW_FIELDS",
    "PullRequests",
    "PullRequestsLike",
    "RECONCILE_ISSUE_FIELDS",
    "RECONCILE_PR_FIELDS",
    "RUN_LIST_FIELDS",
    "RepoMeta",
    "RepoMetaLike",
    "Transport",
    "_ADMIN_FLAG",
    "_LIST_LIMIT",
    "_STRATEGY_FLAGS",
    "_is_mutating",
    "_job_id_from_link",
    "_pr_number_from_url",
    "get_github_issue_dependencies",
]
