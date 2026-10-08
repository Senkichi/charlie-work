"""Derive workflow run ids from check-run links (issue #2540).

A same-name context that a merged Actions job posts via
``POST /repos/{owner}/{repo}/check-runs`` carries the app-assigned
``details_url`` ``https://github.com/<owner>/<repo>/runs/<check_run_id>`` --
GitHub ignores the caller-supplied value for Actions-app check runs -- so
``_run_id_from_link`` (the Actions job-link parser in ``checks.py``) returns
``None`` for it. This module holds the pure half of the fallback derivation:
parse the check-run id out of that link, then resolve the containing
workflow run from a fetched check-run payload via two evidence sources, in
cost order:

1. the caller-supplied ``output`` text -- a posting step that embeds the
   real Actions job URL there gets a one-call resolution;
2. ``check_suite.id`` matched against the ``check_suite_id`` field of
   workflow-run objects -- a check run the Actions app creates from inside
   a workflow job is attached to that run's check suite.

All helpers are data-in/data-out and never raise: any unparseable link,
malformed payload, failed fetch, or matchless suite degrades to ``None``,
which callers treat as "unresolvable" and keep the pre-#2540
definitive-on-first-failure behavior. The I/O half that fetches the check
run and the suite's runs lives in
``charlie_work.orchestration.check_run_ids`` (app-shaped delegates); this
module must stay I/O-free like the rest of the check-classification family.
"""

from __future__ import annotations

import re
from typing import Any

from .checks import _ACTIONS_RUN_LINK_RE

# Matches the check-run-id segment of a GitHub check-run details link, e.g.
# https://github.com/OWNER/REPO/runs/CHECK_RUN_ID (optionally followed by a
# query string or #fragment). GitHub assigns this ``details_url`` shape to
# check runs created by the Actions app -- including the same-name contexts
# a merged Actions job posts via ``POST /repos/{owner}/{repo}/check-runs``,
# whose caller-supplied ``details_url`` GitHub ignores (issue #2540). The
# end anchor keeps real Actions job links from matching: in
# ``/actions/runs/<id>/job/<id>`` the ``/runs/<id>`` segment is followed by
# ``/job/``, never by the end of the URL path.
_CHECK_RUN_LINK_RE = re.compile(r"/runs/(\d+)(?:[?#].*)?$")


def _check_run_id_from_link(link: str | None) -> int | None:
    """Derive a GitHub check-run id from a check's ``/runs/<id>`` details link.

    Counterpart to ``_run_id_from_link`` for the ``.../runs/<check_run_id>``
    details URL GitHub rewrites onto Actions-app check runs (issue #2540).
    The parsed id is the check run's own, NOT a workflow run id, so it must
    be resolved through the Checks API (see ``_run_id_from_check_run_output``
    and ``_run_id_for_check_suite``) before any rerun/log consumer can use
    it. Never raises -- returns ``None`` for anything that doesn't match,
    including Actions job links (those belong to ``_run_id_from_link``).
    """
    if not link:
        return None
    match = _CHECK_RUN_LINK_RE.search(link)
    if not match:
        return None
    return int(match.group(1))


def _run_id_from_check_run_output(payload: dict[str, Any] | None) -> int | None:
    """Derive a workflow run id from a check run's caller-supplied ``output`` text.

    The step that posts a check run via ``POST /repos/{owner}/{repo}/check-runs``
    controls ``output.title``/``summary``/``text`` and may embed the real
    Actions job URL there -- the only URL shape that carries the workflow
    run id, since GitHub replaces the caller's ``details_url`` on Actions-app
    check runs with ``.../runs/<check_run_id>`` (issue #2540) but never
    rewrites the output text. Scanned with the same ``_ACTIONS_RUN_LINK_RE``
    the direct-link path uses, so both paths agree on what counts as a
    parseable job link. Never raises -- returns ``None`` for a missing or
    malformed payload or when no field carries an Actions job link.
    """
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    for key in ("summary", "text", "title"):
        value = output.get(key)
        if not isinstance(value, str):
            continue
        match = _ACTIONS_RUN_LINK_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _check_suite_id_from_payload(payload: dict[str, Any] | None) -> int | None:
    """Extract ``check_suite.id`` from a check-run payload, or ``None``.

    ``GET /repos/{owner}/{repo}/check-runs/{id}`` returns ``check_suite`` as
    an object-or-null carrying a required integer ``id`` (REST Checks API,
    ``Get a check run``). For a check run the Actions app created from
    inside a workflow job, that suite is the workflow run's own check suite
    (issue #2540). Never raises.
    """
    if not isinstance(payload, dict):
        return None
    suite = payload.get("check_suite")
    if not isinstance(suite, dict):
        return None
    suite_id = suite.get("id")
    if isinstance(suite_id, bool) or not isinstance(suite_id, int):
        return None
    return suite_id


def _run_id_for_check_suite(
    workflow_runs: list[dict[str, Any]] | None, check_suite_id: int
) -> int | None:
    """Match a check-suite id against ``workflow_runs`` entries' ``check_suite_id``.

    Every workflow-run object carries the id of the check suite it created
    (REST Actions API, ``Workflow Run`` object field ``check_suite_id``), and
    a workflow run and its check suite correspond one-to-one -- so the entry
    whose ``check_suite_id`` equals the check run's suite id is the
    containing run. Returns ``None`` when no entry matches (the check run
    was NOT created by the Actions app from inside a workflow job -- e.g. an
    external CI posting under its own app -- so no workflow run owns its
    suite) or when the list is unavailable/malformed. Never raises.
    """
    if not isinstance(workflow_runs, list):
        return None
    for run in workflow_runs:
        if not isinstance(run, dict):
            continue
        run_suite_id = run.get("check_suite_id")
        if isinstance(run_suite_id, bool) or not isinstance(run_suite_id, int):
            continue
        if run_suite_id != check_suite_id:
            continue
        run_id = run.get("id")
        if isinstance(run_id, int) and not isinstance(run_id, bool):
            return run_id
    return None
