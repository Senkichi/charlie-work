"""Drill-down routing: path parsing, validation, model call and page render for one request.

``HANDLERS`` is the server side of the route registry (``pages.routes.ROUTES``):
``tests/test_dashboard_drill_server.py`` holds the two equal, so a link renders only for a
route that really has a handler. Path segments are percent-decoded one at a time, after
splitting, so an encoded ``/`` can never smuggle an extra segment into a slug; every value is
then validated by the drill model before anything is opened. Errors come back as house-style
pages (404 for bad or unknown input, 503 for an unreadable source), never as exceptions.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from . import drill
from .drill.flow_stage import stage_listing
from .pages import routes
from .pages.drill_flow import render_flow_stage
from .pages.drill_issue import render_issue
from .pages.drill_pass import render_pass
from .pages.drill_repo import render_repo
from .pages.drill_shell import error_page, not_found_page, status_for
from .read_model import ModelState
from .sources import RepoSource

Response = tuple[int, str]
_NUMBER = re.compile(r"[1-9][0-9]{0,9}")


@dataclass(frozen=True)
class DrillContext:
    state: ModelState
    history_db: Path | None
    repos: Callable[[], Sequence[RepoSource]]
    now: datetime


def _db(ctx: DrillContext) -> Path:
    # No configured dashboard.db reads as "missing" in every model (an error value).
    return ctx.history_db if ctx.history_db is not None else Path("dashboard.db-not-configured")


def _param(params: Mapping[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _fail(ctx: DrillContext, err: drill.DrillError) -> Response:
    return status_for(err), error_page(ctx.state, err)


def _repo(segs: list[str], params: Mapping[str, list[str]], ctx: DrillContext) -> Response:
    if len(segs) != 2:
        return 404, not_found_page(ctx.state, "expected /repo/<owner>/<name>")
    got = drill.repo_drill("/".join(segs), ctx.state.model, _db(ctx))
    if isinstance(got, drill.DrillError):
        return _fail(ctx, got)
    return 200, render_repo(
        ctx.state, got, stage=_param(params, "stage"), view=_param(params, "view")
    )


def _item(kind: str) -> Callable[[list[str], Mapping[str, list[str]], DrillContext], Response]:
    build = drill.issue_drill if kind == "issue" else drill.pr_drill

    def handle(segs: list[str], _params: Mapping[str, list[str]], ctx: DrillContext) -> Response:
        if len(segs) != 3 or _NUMBER.fullmatch(segs[2]) is None:
            return 404, not_found_page(ctx.state, f"expected /{kind}/<owner>/<name>/<number>")
        repo, number = "/".join(segs[:2]), int(segs[2])
        sources = ctx.state.sources
        read = next((r for r in sources.repos if r.key == repo), None) if sources else None
        labels = sources.labels if sources else None
        got = build(repo, number, _db(ctx), read.snapshot if read else None, ctx.now,
                    labels=labels)  # fmt: skip
        if isinstance(got, drill.DrillError):
            return _fail(ctx, got)
        if not got.known:
            noun = "issue" if kind == "issue" else "PR"
            return 404, not_found_page(ctx.state, f"no {noun} #{number} in {repo}'s history")
        return 200, render_issue(ctx.state, got, ctx.now)

    return handle


def _pass(segs: list[str], _params: Mapping[str, list[str]], ctx: DrillContext) -> Response:
    if len(segs) != 3:
        return 404, not_found_page(ctx.state, "expected /pass/<owner>/<name>/<correlation id>")
    got = drill.pass_drill("/".join(segs[:2]), segs[2], ctx.repos())
    if isinstance(got, drill.DrillError):
        return _fail(ctx, got)
    return 200, render_pass(ctx.state, got)


def _flow(segs: list[str], params: Mapping[str, list[str]], ctx: DrillContext) -> Response:
    if len(segs) != 1:
        return 404, not_found_page(ctx.state, "expected /flow/<stage>")
    got = stage_listing(
        segs[0],
        ctx.state.model,
        ctx.state.sources,
        ctx.history_db,
        ctx.now,
        repo=_param(params, "repo"),
    )
    if isinstance(got, drill.DrillError):
        return _fail(ctx, got)
    return 200, render_flow_stage(ctx.state, got)


HANDLERS: Mapping[str, Callable[[list[str], Mapping[str, list[str]], DrillContext], Response]] = {
    "/repo": _repo,
    "/issue": _item("issue"),
    "/pr": _item("pr"),
    "/pass": _pass,
    "/flow": _flow,
}


def drill_response(
    path: str, params: Mapping[str, list[str]], ctx: DrillContext
) -> Response | None:
    """The response for a drill-down path, or ``None`` when ``path`` is not one."""
    head, _, rest = path.lstrip("/").partition("/")
    prefix = "/" + head
    handler = HANDLERS.get(prefix)
    if handler is None or not routes.is_routed(prefix):
        return None
    segs = [unquote(s) for s in rest.split("/")] if rest else []
    if any(not s for s in segs):
        return 404, not_found_page(ctx.state, "empty path segment")
    return handler(segs, params, ctx)
