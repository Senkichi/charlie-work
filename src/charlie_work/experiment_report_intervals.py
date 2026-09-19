"""Interval math for the per-PR experiment read-out (issue #1701).

Extracted from ``experiment_report.py`` under the repo's 800-line module
cap.  Two interval shapes are needed:

* a per-arm binomial rate interval -- :func:`wilson_interval`;
* a per-pair *difference* interval -- :func:`rate_difference_interval`,
  the Newcombe hybrid-score method (Newcombe 1998, method 10), which
  combines the two Wilson intervals rather than approximating the
  difference directly.

The difference interval must not collapse at zero observed events.  The
Wald/normal approximation it replaces returns a zero-width ``[0, 0]``
interval at ``k1 == k2 == 0`` no matter how small the evidence is, which
let the stopping rule read "no detectable difference" off literally no
outcome data.  Newcombe's bounds still tighten with *n*, but they never
degenerate to a point: at ``0/200 vs 0/200`` the interval is roughly
``[-0.019, +0.019]``, an honest statement about what the data can rule
out -- and the stopping rule additionally refuses equivalence below a
minimum observed-event floor (see ``experiment_report_stopping``).
"""

from __future__ import annotations

import math

CONFIDENCE_Z = 1.96


def wilson_interval(k: int, n: int, z: float = CONFIDENCE_Z) -> tuple[float, float] | None:
    """Wilson score interval for a binomial rate; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_difference_interval(
    k1: int, n1: int, k2: int, n2: int, z: float = CONFIDENCE_Z
) -> tuple[float, float] | None:
    """Newcombe hybrid-score interval for the difference of two rates (a - b).

    The issue asks for "the difference between arms for each rate with an
    interval"; a difference interval (unlike two CIs compared visually)
    directly answers "is the gap distinguishable from zero".  Method 10 of
    Newcombe (1998): subtract the Wilson bounds of the opposite arms under
    a square-and-root combination, which stays informative at boundary
    counts -- unlike the Wald interval, it cannot collapse to ``[0, 0]``
    when both arms observe zero events.  Clamped to [-1, 1]; None when
    either denominator is 0.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    w1 = wilson_interval(k1, n1, z)
    w2 = wilson_interval(k2, n2, z)
    assert w1 is not None and w2 is not None  # n1, n2 > 0
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - w1[0]) ** 2 + (w2[1] - p2) ** 2)
    hi = diff + math.sqrt((w1[1] - p1) ** 2 + (p2 - w2[0]) ** 2)
    return (max(-1.0, lo), min(1.0, hi))
