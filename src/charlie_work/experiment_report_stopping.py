"""The experiment's stopping rule, evaluated against one report window
(issue #1701).

Extracted from ``experiment_report.py`` under the repo's 800-line module
cap.  The rule is documented in docs/review-effort-experiment.md; this
module is its only implementation.

The rule must never report ``met`` on absent evidence.  Two guards make
that structural rather than a matter of wording:

* **A stated equivalence margin.**  ``no_detectable_difference`` requires
  every pairwise outcome-difference interval to lie entirely inside
  ``[-EQUIVALENCE_MARGIN, +EQUIVALENCE_MARGIN]`` -- a positive claim that
  any remaining gap is too small to matter, not merely "the interval
  spans 0" (a wide interval spans 0 and proves nothing).
* **A minimum observed-outcome floor.**  Every arm must have produced at
  least ``MIN_EQUIVALENCE_EVENTS_PER_ARM`` outcome events.  With zero
  outcome events the interval math cannot distinguish "the arms are
  equivalent" from "the outcome channel never fires in this pipeline" --
  ending the experiment there would declare victory for an instrument
  that may simply be dead.  Below the floor the rule reports ``not_met``
  (underpowered), never equivalence.
"""

from __future__ import annotations

from typing import Any

# Stopping-rule defaults.  These match docs/review-effort-experiment.md;
# --min-prs-per-arm overrides the minimum for ad-hoc analysis, not the rule.
DEFAULT_MIN_PRS_PER_ARM = 100
# Every arm must reach this multiple of the minimum assigned-PR count before
# equivalence can be declared at all.
EQUIVALENCE_MULTIPLE = 2
# Stated equivalence margin: a pairwise outcome-difference interval must lie
# entirely inside +/- this absolute rate difference for the "no detectable
# difference" verdict -- otherwise the data cannot rule out a gap that size.
EQUIVALENCE_MARGIN = 0.05
# Minimum observed outcome events per arm for equivalence.  Zero observed
# events cannot distinguish equivalent arms from a dead outcome channel;
# the floor also keeps the interval math honest at boundary counts.
MIN_EQUIVALENCE_EVENTS_PER_ARM = 1


def evaluate_stopping_rule(
    arms: list[str],
    prs_assigned: dict[str, int],
    outcome_metric: dict[str, Any] | None,
    min_prs_per_arm: int,
    z: float,
) -> dict[str, Any]:
    """The experiment's stopping rule, evaluated against this window's data.

    The rule (docs/review-effort-experiment.md):
    * every arm needs >= ``min_prs_per_arm`` assigned PRs before anything is
      decided;
    * once that holds, an outcome-difference interval that excludes 0 ends
      the experiment in favour of the lower-defect arm;
    * every arm at >= ``EQUIVALENCE_MULTIPLE`` x the minimum ends it as
      "no detectable difference" only when every pairwise difference
      interval sits entirely inside +/- ``EQUIVALENCE_MARGIN`` AND every
      arm observed at least ``MIN_EQUIVALENCE_EVENTS_PER_ARM`` outcome
      events;
    * anything else -- including a one-arm or zero-arm report, or a window
      with no derivable outcome metric -- is ``"not_met"``.

    ``outcome_metric`` is the report's outcome metric dict (the one built
    by ``build_report`` for ``post_approval_defect_rate``), or ``None``
    when no outcome metric is derivable from the recorded events -- in
    which case the rule can never be met on activity metrics alone.
    """
    equivalence_min = EQUIVALENCE_MULTIPLE * min_prs_per_arm
    rule: dict[str, Any] = {
        "min_prs_per_arm": min_prs_per_arm,
        "equivalence_min_prs_per_arm": equivalence_min,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "min_equivalence_events_per_arm": MIN_EQUIVALENCE_EVENTS_PER_ARM,
        "outcome_metric": outcome_metric["name"] if outcome_metric else None,
        "confidence_z": z,
        "post_experiment_config": "reviewer.effort_experiment_fraction: 0.0",
        "met": False,
        "verdict": "not_met",
        "detail": "",
    }
    if len(arms) < 2:
        rule["detail"] = f"fewer than two arms observed ({len(arms)}); nothing to compare"
        return rule
    if outcome_metric is None:
        rule["detail"] = (
            "no outcome metric is derivable from the recorded event stream; "
            "the experiment cannot end on activity metrics alone"
        )
        return rule
    short = {a: n for a, n in prs_assigned.items() if n < min_prs_per_arm}
    if short:
        detail = ", ".join(f'arm "{a}" has {n} assigned PRs' for a, n in sorted(short.items()))
        rule["detail"] = f"minimum {min_prs_per_arm} assigned PRs per arm not reached: {detail}"
        return rule
    decided = [
        d
        for d in outcome_metric["differences"]
        if d["ci95"] and (d["ci95"][0] > 0 or d["ci95"][1] < 0)
    ]
    if decided:
        best = decided[0]
        a, b = best["pair"]
        # diff = rate_a - rate_b; a strictly positive interval means a is worse.
        better = b if best["diff"] > 0 else a
        worse = a if best["diff"] > 0 else b
        rule.update(
            met=True,
            verdict="difference_detected",
            detail=(
                f'{outcome_metric["name"]} differs: arm "{better}" is lower '
                f"(pair {a} vs {b}, diff {best['diff']:+.4f}, "
                f"95% CI [{best['ci95'][0]:+.4f}, {best['ci95'][1]:+.4f}]). "
                f'Adopt arm "{better}" (lower post-approval defect rate over '
                f'arm "{worse}") and set reviewer.effort_experiment_fraction: 0.0.'
            ),
        )
        return rule
    if not all(n >= equivalence_min for n in prs_assigned.values()):
        rule["detail"] = (
            "minimum per-arm N reached, but the outcome-metric difference "
            "intervals are not yet decisive and the equivalence bound "
            f"({equivalence_min} PRs/arm) is not reached -- keep running."
        )
        return rule
    # Equivalence bound reached: equivalence needs both a positive margin
    # claim and a minimum of observed outcome evidence.
    events_per_arm = {a: (outcome_metric["per_arm"].get(a) or {}).get("k") for a in arms}
    starved = {
        a: k
        for a, k in events_per_arm.items()
        if not isinstance(k, int) or k < MIN_EQUIVALENCE_EVENTS_PER_ARM
    }
    if starved:
        detail = ", ".join(
            f'arm "{a}" observed {k if isinstance(k, int) else 0} outcome events'
            for a, k in sorted(starved.items())
        )
        rule["detail"] = (
            f"equivalence bound ({equivalence_min} PRs/arm) reached but the "
            f"rule is underpowered: {detail} -- every arm must observe at "
            f"least {MIN_EQUIVALENCE_EVENTS_PER_ARM} outcome event(s) before "
            "the intervals can distinguish equivalent arms from a dead "
            "outcome channel -- keep running."
        )
        return rule
    all_pairs_intervalled = bool(outcome_metric["differences"]) and all(
        d["ci95"] for d in outcome_metric["differences"]
    )
    within_margin = all_pairs_intervalled and all(
        d["ci95"][0] >= -EQUIVALENCE_MARGIN and d["ci95"][1] <= EQUIVALENCE_MARGIN
        for d in outcome_metric["differences"]
    )
    if within_margin:
        rule.update(
            met=True,
            verdict="no_detectable_difference",
            detail=(
                f"every arm has >= {equivalence_min} assigned PRs, every arm "
                f"observed >= {MIN_EQUIVALENCE_EVENTS_PER_ARM} outcome "
                f"event(s), and every {outcome_metric['name']} difference "
                f"interval lies inside the stated equivalence margin "
                f"(+/-{EQUIVALENCE_MARGIN:.2f}). End the experiment "
                "(effort_experiment_fraction: 0.0); pick the arm with the "
                "lower review_cost_per_pr_usd_mean."
            ),
        )
        return rule
    rule["detail"] = (
        f"equivalence bound ({equivalence_min} PRs/arm) reached, but the "
        f"{outcome_metric['name']} difference intervals are wider than the "
        f"stated equivalence margin (+/-{EQUIVALENCE_MARGIN:.2f}) without "
        "excluding 0 -- keep running."
    )
    return rule
