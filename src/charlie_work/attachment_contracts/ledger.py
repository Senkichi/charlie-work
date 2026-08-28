"""Linear-ledger detection: structural exemption for monotonic numbered families.

No allowlist of module/class names. A member family is a "ledger" purely by its
own naming shape: a dominant `<prefix><int>` pattern with strictly increasing,
near-contiguous integers. See spec section "Ledger detection".
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

# `^(?P<prefix>[A-Za-z_]+?)(?P<n>\d+)$` — non-greedy prefix, trailing integer.
# Digits cannot appear in the prefix (character class excludes them), so the
# regex only ever matches names whose *trailing* run is numeric.
_LEDGER_MEMBER_RE = re.compile(r"^(?P<prefix>[A-Za-z_]+?)(?P<n>\d+)$")

# Statistical/structural floors, not size thresholds:
MIN_MATCHING_MEMBERS = 3
DOMINANCE_THRESHOLD = 0.8
# "strictly increasing and contiguous-modulo-gaps<=1": consecutive integers in
# the dominant-prefix sequence may differ by 1 (contiguous) or 2 (one skipped
# number tolerated); anything else (duplicate, decrease, or a bigger gap)
# breaks the ledger pattern.
_MAX_STEP = 2


def classify_ledger(members: Sequence[str]) -> bool:
    """True iff `members` forms a linear-ledger family.

    Requires >= MIN_MATCHING_MEMBERS names matching `<prefix><int>`, one
    dominant prefix covering >= DOMINANCE_THRESHOLD of ALL members (so a
    non-matching or off-pattern member dilutes dominance), and that dominant
    prefix's integers strictly increasing with gaps of at most one skipped
    number.
    """
    total = len(members)
    if total == 0:
        return False

    matches: list[tuple[str, int]] = []
    for name in members:
        m = _LEDGER_MEMBER_RE.match(name)
        if m is not None:
            matches.append((m.group("prefix"), int(m.group("n"))))

    if len(matches) < MIN_MATCHING_MEMBERS:
        return False

    prefix_counts = Counter(prefix for prefix, _ in matches)
    dominant_prefix, dominant_count = prefix_counts.most_common(1)[0]
    if dominant_count / total < DOMINANCE_THRESHOLD:
        return False

    numbers = sorted(n for prefix, n in matches if prefix == dominant_prefix)
    if len(numbers) < MIN_MATCHING_MEMBERS:
        return False

    for prev, curr in zip(numbers, numbers[1:]):
        step = curr - prev
        if step < 1 or step > _MAX_STEP:
            return False

    return True
