"""Transport capability: shared low-level GitHub CLI/HTTP plumbing (#1585).

Not a ``GitHubLike`` sub-protocol cluster. Transport is the destination for
the 13 non-protocol internals (design doc Section 3.2: ``_run_bool``,
``_list_json``, ``_graphql_query``, the retry-knob helpers, etc.) that are
call targets from every other cluster but are not themselves part of the
public ``GitHubLike`` surface. ``run`` and ``__post_init__`` stay on the
owner (``GitHub``) as the interception seam and dataclass hook respectively
-- they never move here.
"""

from __future__ import annotations

from ._base import CapabilityCollaborator


class Transport(CapabilityCollaborator):
    """Shared low-level transport capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
