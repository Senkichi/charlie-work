"""Stage drill-down page (``/flow/<stage>``): the issues behind one Flow number on Now."""

from __future__ import annotations

from datetime import tzinfo

from ..drill.flow_stage import DONE, StageListing
from ..read_model import ModelState
from .drill_shell import page, section, table, ts_tag
from .now_fmt import esc, issue_url, link, local_time, pr_url, repo_url, short_repo


def _repo_cell(repo: str) -> str:
    return link(repo_url(repo), short_repo(repo), "repo", repo)


def _issue_rows(s: StageListing) -> list[str]:
    return [
        f"<tr><td>{_repo_cell(i.repo)}</td>"
        f'<td class="num">{link(issue_url(i.repo, i.number), f"#{i.number}")}</td>'
        f"<td>{esc(i.title or '(untitled)')}</td>"
        f'<td class="labels">{esc(", ".join(i.labels)) or "—"}</td></tr>'
        for i in s.items
    ]


def _merge_rows(s: StageListing, tz: tzinfo | None) -> list[str]:
    out = []
    for repo, m in s.merges:
        item = " · ".join(
            x
            for x in (
                link(issue_url(repo, m.issue), f"#{m.issue}") if m.issue else "",
                link(pr_url(repo, m.pr), f"PR #{m.pr}") if m.pr else "",
            )
            if x
        )
        approx = ' <span class="aprx">approx.</span>' if m.approx else ""
        out.append(
            f"<tr><td>{ts_tag(m.ts, tz)}</td><td>{_repo_cell(repo)}</td>"
            f"<td>{item or '—'}</td><td><code>{esc(m.evidence)}</code>{approx}</td></tr>"
        )
    return out


def render_flow_stage(state: ModelState, s: StageListing, tz: tzinfo | None = None) -> str:
    scope = f" in {short_repo(s.repo)}" if s.repo else " across the fleet"
    count = "unknown" if s.count is None else str(s.count)
    if s.key == DONE:
        lead = (
            f"Now shows <b>{esc(count)}</b> merged in the last 24h{esc(scope)}. "
            f"Read from dashboard.db at {local_time(s.as_of)} (local)."
            if s.as_of
            else ""
        )
        inner = table(
            "Merges in the last 24 hours",
            ("Merged (local)", "Repo", "Item", "Evidence"),
            _merge_rows(s, tz),
            "No merges recorded in the last 24 hours.",
        )
    else:
        when = f" Snapshot tick {local_time(s.as_of)} (local)." if s.as_of else ""
        lead = f"Now shows <b>{esc(count)}</b>{esc(scope)}, as of snapshot.{when}"
        inner = table(
            f"Issues in {s.name}",
            ("Repo", "Issue", "Title", "Labels"),
            _issue_rows(s),
            f"No issues in {s.name}{scope}.",
        )
    notes = ""
    if s.note:
        notes += f'<p class="dnote">{esc(s.note)}</p>'
    if s.unreadable:
        notes += (
            '<p class="dnote text-warn">Not counted (snapshot unreadable): '
            f"{esc(', '.join(s.unreadable))}</p>"
        )
    body = f'<p class="dlead">{lead}</p>{notes}' + section("items", s.name, inner)
    trail = [(s.name, None)]
    if s.repo:
        trail.insert(0, (short_repo(s.repo), repo_url(s.repo)))
    return page(state, "Flow", f"{s.name}{scope}", trail, body)
