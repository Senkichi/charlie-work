"""Line-content half of a Verdict's carry-forward check (issues #414, #1187, #3090).

``git patch-id --stable`` strips leading whitespace from ``+``/``-`` lines, so
an indentation-only change -- which changes what Python does -- keeps the same
patch-id. Every carry-forward therefore also compares the whitespace-exact
signature stored with the Verdict. The PR check (``_check_carry_forward``) and
the Local path's ``approval_survives_head_move`` share this one comparison;
what a record with no stored signature means is each caller's decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from charlie_work.janitor import DiffContentSignature


def stored_signature_matches(
    decision: Mapping[str, Any], live_signature: DiffContentSignature
) -> bool | None:
    """Whether ``decision``'s stored line signature equals ``live_signature``.

    ``None`` when the decision stores no signature (either field absent), so a
    caller cannot mistake "nothing to compare" for a match or a mismatch.
    Lines compare in order; files compare as a set.
    """
    reviewed_changed_lines = decision.get("reviewed_changed_lines")
    reviewed_changed_files = decision.get("reviewed_changed_files")
    if reviewed_changed_lines is None or reviewed_changed_files is None:
        return None
    return (
        tuple(reviewed_changed_lines) == live_signature.changed_lines
        and frozenset(reviewed_changed_files) == live_signature.changed_files
    )
