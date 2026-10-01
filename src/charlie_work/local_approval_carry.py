"""Patch-id carry-forward of a local-lane approval across a head move (issue #2151).

The merge gate's own base-sync merge advances the branch head without changing
the reviewed delta. Both the gate and the packet phase must agree that such a
move keeps the approval; this is the one predicate they share.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.janitor import _calculate_patch_id
from charlie_work.local_lane import branch_diff


def approval_survives_head_move(
    repo_root: Path,
    base_ref: str,
    branch: str,
    decision: dict[str, Any],
) -> bool:
    """True when the live branch diff is patch-id-equal to the approved head's.

    Compares against the stored ``reviewed_patch_id`` first. When that does not
    match -- or is missing, as on verdicts recorded before #2151 pinned it to
    the reviewed head -- the patch-id is re-derived from ``reviewed_head_sha``
    itself, so a mis-recorded baseline cannot void an approval whose content is
    unchanged. A genuinely different diff matches neither and returns False.
    """
    live_diff = branch_diff(repo_root, base_ref, branch)
    if not live_diff:
        return False
    live_patch = _calculate_patch_id(live_diff)
    if not live_patch:
        return False
    if live_patch == decision.get("reviewed_patch_id"):
        return True
    reviewed_head = decision.get("reviewed_head_sha")
    if not reviewed_head:
        return False
    reviewed_diff = branch_diff(repo_root, base_ref, str(reviewed_head))
    return bool(reviewed_diff) and _calculate_patch_id(reviewed_diff) == live_patch
