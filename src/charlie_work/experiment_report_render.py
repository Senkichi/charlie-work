"""Text rendering for the per-PR experiment read-out (issue #1701).

Extracted from ``experiment_report.py`` under the repo's 800-line module
cap.  :func:`render_text` turns the report dict produced by
:func:`charlie_work.experiment_report.build_report` into the command's
human-readable output.  The JSON shape is the report dict itself; this
module only formats it, so a rendering change never alters the data.
"""

from __future__ import annotations

from typing import Any, Mapping

from .experiment_report_scan import NO_OUTCOME_STATEMENT


def _fmt_rate(entry: Mapping[str, Any]) -> str:
    if entry["n"] == 0 or entry["rate"] is None:
        return f"0/{entry['n']} = n/a (no PRs in denominator)"
    ci = entry["ci95"]
    ci_s = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    return f"{entry['k']}/{entry['n']} = {entry['rate']:.3f}{ci_s}"


def _fmt_value(entry: Mapping[str, Any]) -> str:
    if entry["value"] is None:
        return f"n/a (n={entry['n']})"
    return f"{entry['value']:.3f} (n={entry['n']})"


def render_text(report: Mapping[str, Any]) -> str:
    """Render the report dict as the command's human-readable output."""
    lines: list[str] = []
    w = report["window"]
    window_bits = []
    if w["since"]:
        window_bits.append(f"since={w['since']}")
    if w["until"]:
        window_bits.append(f"until={w['until']}")
    for s, e in w["exclude_windows"]:
        window_bits.append(f"exclude=[{s} .. {e}]")
    lines.append(
        f'experiment report: session-metrics key "{report["metrics_key"]}" '
        "(read-only; no state was modified)"
    )
    lines.append(
        f"window: {', '.join(window_bits) if window_bits else 'unfiltered'}; "
        f"events scanned: {report['events_scanned']}"
    )
    if report.get("events_db"):
        lines.append(f"source: {report['events_db']}")
    lines.append(
        "unit of analysis: the PR (arm assignment is stable per PR across "
        "rounds; rates count each PR once)"
    )
    if not report["arms"]:
        lines.append(
            f'NO DATA: no PR carries a "{report["metrics_key"]}" value in the scanned window.'
        )
    for arm in report["arms"]:
        pa = report["per_arm"][arm]
        lines.append(
            f'arm "{arm}": {pa["prs_assigned"]} assigned, '
            f"{pa['prs_with_rounds']} with recorded rounds"
        )
    lines.append("")
    lines.append("metrics:")
    for name, m in report["metrics"].items():
        lines.append(f"  {name} [{m['measure']}]")
        lines.append(f"    {m['description']}")
        for arm in report["arms"]:
            entry = m["per_arm"].get(arm)
            if entry is None:
                continue
            if m["kind"] == "rate":
                rendered = _fmt_rate(entry)
            elif m["kind"] == "count":
                rendered = str(entry["n"])
            else:
                rendered = _fmt_value(entry)
            extra = ""
            if name == "review_cost_per_pr_usd_mean" and "rounds_total" in entry:
                extra = (
                    f"  (cost data on {entry['rounds_with_cost']}/{entry['rounds_total']} rounds)"
                )
            lines.append(f"    {arm}: {rendered}{extra}")
        for d in m["differences"]:
            a, b = d["pair"]
            lines.append(
                f"    diff {a} - {b} = {d['diff']:+.4f}  "
                f"95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            )
    lines.append("")
    lines.append("outcome coverage:")
    derivable = report["outcome_coverage"]["derivable"]
    if derivable:
        lines.append("  derivable and reported above: " + ", ".join(derivable))
    else:
        lines.append(f"  {report['outcome_coverage'].get('statement') or NO_OUTCOME_STATEMENT}")
    for item in report["outcome_coverage"]["not_derivable"]:
        lines.append(
            f"  NOT derivable from recorded events: {item['candidate']} ({item['reason']})"
        )
    warnings = report["data_integrity_warnings"]
    if warnings or report["prs_with_rounds_without_arm_value"]:
        lines.append("")
        lines.append("data integrity:")
        for wrn in warnings:
            lines.append(f"  PR #{wrn['pr']}: {wrn['detail']}")
        n = report["prs_with_rounds_without_arm_value"]
        if n:
            lines.append(
                f"  {n} PR(s) with recorded rounds carried no "
                f'"{report["metrics_key"]}" value (outside the experiment)'
            )
    lines.append("")
    rule = report["stopping_rule"]
    status = "MET" if rule["met"] else "NOT MET"
    lines.append(f"stopping rule: {status} ({rule['verdict']})")
    lines.append(
        f"  min PRs/arm: {rule['min_prs_per_arm']} "
        f"(equivalence bound: {rule['equivalence_min_prs_per_arm']}/arm, "
        f"equivalence margin: +/-{rule['equivalence_margin']:.2f}, "
        f"min outcome events/arm: {rule['min_equivalence_events_per_arm']}); "
        f"outcome metric: {rule['outcome_metric'] or 'none derivable'}"
    )
    lines.append(f"  {rule['detail']}")
    lines.append(
        f"  when met: set {rule['post_experiment_config']} (docs/review-effort-experiment.md)"
    )
    return "\n".join(lines)
