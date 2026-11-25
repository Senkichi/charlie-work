"""Patch-id carry-forward of a local-lane approval across a head move (issue #2151).

The merge gate's own base-sync merge advances the branch head without changing
the reviewed delta. Both the gate and the packet phase must agree that such a
move keeps the approval; this is the one predicate they share.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.carry_forward_signature import stored_signature_matches
from charlie_work.janitor import _calculate_patch_id, _diff_content_signature
from charlie_work.local_lane import branch_diff


def approval_survives_head_move(
    repo_root: Path,
    base_ref: str,
    branch: str,
    decision: dict[str, Any],
) -> bool:
    """True when the live branch diff carries the content the approval reviewed.

    Compares against the stored ``reviewed_patch_id`` first. When that does not
    match -- or is missing, as on verdicts recorded before #2151 pinned it to
    the reviewed head -- the patch-id is re-derived from ``reviewed_head_sha``
    itself, so a mis-recorded baseline cannot void an approval whose content is
    unchanged. A genuinely different diff matches neither and returns False.

    Issue #3090: a patch-id match is necessary, not sufficient. ``git patch-id
    --stable`` strips leading whitespace, so an indentation-only change --
    which changes what Python does -- keeps the same id. The live diff must
    also equal the whitespace-exact line signature stored with the verdict
    (``reviewed_changed_lines`` / ``reviewed_changed_files``), on both patch-id
    routes above. A record that stores no signature has nothing to compare
    line content against and never carries: it is reviewed again. (The PR
    check, ``_check_carry_forward``, shares the comparison but still carries
    such a record on the patch-id alone.)

    Issue #2739: ``branch_diff``'s three outcomes carry different semantics
    here -- ``None`` is a failed diff (carries nothing), ``""`` is the
    legitimate "already landed" answer (the branch's whole content is on the
    base), and anything else is real content to compare by patch-id. An
    empty live diff carries an approval only when the verdict was recorded
    over that same empty shape (``reviewed_patch_id == ""``); a verdict
    recorded over real content whose diff later empties is the terminal
    landed path's job, not this predicate's.
    """
    live_diff = branch_diff(repo_root, base_ref, branch)
    if live_diff is None:
        return False
    if live_diff == "":
        return decision.get("reviewed_patch_id") == ""
    live_patch = _calculate_patch_id(live_diff)
    if not live_patch:
        return False
    if not stored_signature_matches(decision, _diff_content_signature(live_diff)):
        return False
    if live_patch == decision.get("reviewed_patch_id"):
        return True
    reviewed_head = decision.get("reviewed_head_sha")
    if not reviewed_head:
        return False
    reviewed_diff = branch_diff(repo_root, base_ref, str(reviewed_head))
    return bool(reviewed_diff) and _calculate_patch_id(reviewed_diff) == live_patch
