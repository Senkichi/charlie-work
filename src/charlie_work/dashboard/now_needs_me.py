"""Needs-me rows for the Now model: Operator queue, human-needed PRs, alarms, stale sources.

Commands use only subcommands that exist in ``cli.py`` (``unescalate``, ``fleet resume``);
``tests/test_dashboard_now_model.py`` parses each through ``cli.build_parser``.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .now_access import as_int, dict_list, label_set, snapshot_data
from .now_types import FindingLike, NeedsMeItem, RepoFreshness, RepoRead, SourcesRead
from .sources import _parse_utc

_SEVERITY_RANK = {"anomaly": 0, "action": 1, "warn": 2}


def _cli(repo: RepoRead, *args: str) -> str:
    """A copy-paste ``charlie`` command targeting ``repo`` (``--repo`` is a global flag)."""
    return shlex.join(["charlie", "--repo", repo.repo_root, *args])


def needs_me_items(
    sources: SourcesRead,
    fresh: tuple[RepoFreshness, ...],
    findings: Sequence[FindingLike],
    now: datetime,
    threshold: float,
    runners_stale_age: float | None,
) -> tuple[NeedsMeItem, ...]:
    labels = sources.labels
    items: list[NeedsMeItem] = []
    for repo in sources.repos:
        data = snapshot_data(repo)
        since = dict(repo.escalated_since)
        pr_by_issue = {
            as_int(pr.get("issue_number")): as_int(pr.get("number"))
            for pr in dict_list(data, "prs")
        }

        def age(number: int, _since: dict[int, datetime] = since) -> float | None:
            when = _since.get(number)
            return (now - when).total_seconds() if when is not None else None

        for issue in dict_list(data, "issues"):
            number = as_int(issue.get("number"))
            have = label_set(issue)
            title = str(issue.get("title") or "")
            if labels.operator_queue in have:
                items.append(
                    NeedsMeItem(
                        "operator_queue",
                        "action",
                        repo.key,
                        age(number),
                        f"Operator queue: #{number} {title}".rstrip(),
                        _cli(repo, "unescalate", "--issue", str(number)),
                        True,
                    )
                )
            elif labels.human_needed in have:
                pr = pr_by_issue.get(number)
                if pr:
                    items.append(
                        NeedsMeItem(
                            "human_needed",
                            "action",
                            repo.key,
                            age(number),
                            f"Human needed: PR #{pr} (issue #{number}) awaits an operator verdict",
                            _cli(repo, "unescalate", "--pr", str(pr)),
                            True,
                        )
                    )
                else:
                    items.append(
                        NeedsMeItem(
                            "human_needed",
                            "action",
                            repo.key,
                            age(number),
                            f"Human needed: #{number} {title}".rstrip(),
                            _cli(repo, "unescalate", "--issue", str(number)),
                            True,
                        )
                    )
    for finding in findings:
        if finding.severity in ("anomaly", "warn"):
            items.append(
                NeedsMeItem(
                    "alarm",
                    "anomaly" if finding.severity == "anomaly" else "warn",
                    finding.repo,
                    None,
                    f"{finding.check}: {finding.detail}",
                    None,
                    False,
                )
            )
    for row in fresh:
        if not row.stale:
            continue
        reason = (
            f"snapshot unreadable: {row.error}"
            if row.error
            else f"snapshot is {int(row.age_seconds or 0)}s old (stale after {int(threshold)}s)"
        )
        items.append(
            NeedsMeItem("stale_source", "warn", row.repo, row.age_seconds, reason, None, False)
        )
    if runners_stale_age is not None:
        items.append(
            NeedsMeItem(
                "stale_source",
                "warn",
                "fleet",
                runners_stale_age,
                f"runner_allocation is {int(runners_stale_age)}s old (stale after {int(threshold)}s)",
                None,
                False,
            )
        )
    if sources.pause:
        items.append(
            NeedsMeItem(
                "paused",
                "anomaly",
                "fleet",
                _since_age(sources.pause.get("paused_at"), now),
                f"Fleet paused: {sources.pause.get('reason') or 'no reason recorded'}",
                "charlie fleet resume",
                False,
            )
        )
    hb = sources.supervisor_heartbeat
    if hb is not None and hb.data is not None:
        beat = _since_age(hb.data.get("last_beat_at"), now)
        if hb.data.get("exited_at") or (beat is not None and beat > threshold):
            items.append(
                NeedsMeItem(
                    "supervisor",
                    "anomaly",
                    "fleet",
                    beat,
                    "Supervisor is not beating (exited or last beat too old)",
                    None,
                    False,
                )
            )
    # Anomalies first, then operator actions, then warnings; oldest first within a
    # tier (unknown ages last), key as the deterministic tie-break.
    return tuple(
        sorted(
            items,
            key=lambda i: (
                _SEVERITY_RANK[i.severity],
                -(i.age_seconds if i.age_seconds is not None else -1.0),
                i.repo,
                i.reason,
            ),
        )
    )


def _since_age(value: Any, now: datetime) -> float | None:
    when = _parse_utc(value)
    return (now - when).total_seconds() if when is not None else None
