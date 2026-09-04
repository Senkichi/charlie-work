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

from ._base import GitHubRunResult
from .checks import PR_CHECKS_FIELDS, Checks, ChecksLike, _job_id_from_link
from .comments import Comments, CommentsLike
from .issues import Issues, IssuesLike
from .labels import LABEL_LIST_FIELDS, Labels, LabelsLike
from .merge_branch import MergeBranch, MergeBranchLike
from .pull_requests import PullRequests, PullRequestsLike
from .repo_meta import RepoMeta, RepoMetaLike
from .transport import Transport

__all__ = [
    "Checks",
    "ChecksLike",
    "Comments",
    "CommentsLike",
    "GitHubRunResult",
    "Issues",
    "IssuesLike",
    "LABEL_LIST_FIELDS",
    "Labels",
    "LabelsLike",
    "MergeBranch",
    "MergeBranchLike",
    "PR_CHECKS_FIELDS",
    "PullRequests",
    "PullRequestsLike",
    "RepoMeta",
    "RepoMetaLike",
    "Transport",
    "_job_id_from_link",
]
