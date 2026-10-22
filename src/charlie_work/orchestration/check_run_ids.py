"""Run-id resolution delegates for check-run-linked required checks (#2540).

The I/O half of issue #2540's fix: the data-boundary enrichment that
resolves workflow run ids for required checks whose ``link`` is the
app-assigned check-run shape (``.../runs/<id>``), plus the two-endpoint
resolver it drives. The pure derivation helpers live in
``charlie_work.check_run_resolution`` -- this module performs the ``gh``
I/O and is app-shaped: ``workflow_delegation._install_delegates`` attaches
every top-level ``def`` here onto ``OrchestratorApp`` (both are net-new
members, declared under ``tests/baselines/post_campaign_surface_additions/``).

Called at the same data-boundary points as the #1383 enrichment
(``_enrich_checks_infra_blocked`` in ``orchestration/misc_checks.py``):
before the janitor gate in ``review()`` and in ``merge_ready``'s
``gather_checks``, so both paths see the same resolved run ids.
"""

from __future__ import annotations

from typing import Any

from charlie_work.check_run_resolution import (
    _check_run_id_from_link,
    _check_suite_id_from_payload,
    _run_id_for_check_suite,
    _run_id_from_check_run_output,
)
from charlie_work.checks import _CheckClassification, _classify_check_run


def _enrich_checks_run_ids(
    self, checks: list[dict[str, Any]] | None, required: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Resolve workflow run ids for required checks whose ``link`` is a
    check-run link (``.../runs/<id>``), not an Actions job link (issue #2540).

    A same-name context a merged Actions job posts via
    ``POST /repos/{owner}/{repo}/check-runs`` carries the app-assigned
    ``details_url`` ``https://github.com/<owner>/<repo>/runs/<check_run_id>``
    -- GitHub ignores the caller-supplied ``details_url`` for Actions-app
    check runs -- so ``pr_checks``'s link-derived ``runId`` injection yields
    ``None`` for it. Both debounce classifiers
    (``classify_check_failures``/``classify_infra_failures``) then treat any
    failure of that check as definitive on first sight: no flake-aware
    rerun is ever scheduled and no run id exists for log fetching.

    Same data-boundary shape as ``_enrich_checks_infra_blocked``: kept as an
    ``OrchestratorApp``-shaped delegate (not a pure function in
    ``checks.py``) because resolution needs up to two ``gh`` API calls per
    unresolved check (``check_run`` / ``workflow_runs_for_check_suite``) --
    I/O that the pure classification layer must not perform. Both API
    methods return safe empty values (``None``) on any GitHub failure and
    never raise, so an unresolvable check degrades to the pre-#2540
    definitive-first-failure routing rather than crashing ingestion.

    Cost-bounded by construction: only required checks that are currently
    failing (``FAIL``) or infra-failed (``CANCELLED``/``INFRA``) -- the only
    runs the debounce classifiers need ids for -- with an unparseable run id
    and a parseable check-run link are resolved; passing, pending, skipped,
    non-required, and Actions-job-linked checks pass through untouched, so
    the common fleet pass spends zero extra API calls.
    """
    if not checks or not required:
        return list(checks or [])
    required_set = set(required)
    enriched: list[dict[str, Any]] = []
    for check in checks:
        name = str(check.get("name") or "")
        if name in required_set and not isinstance(check.get("runId"), int):
            classification = _classify_check_run(check)
            if classification in {
                _CheckClassification.FAIL,
                _CheckClassification.CANCELLED,
                _CheckClassification.INFRA,
            }:
                check_run_id = _check_run_id_from_link(check.get("link"))
                if check_run_id is not None:
                    resolved = self._resolve_run_id_via_check_run(check_run_id)
                    if resolved is not None:
                        check = {**check, "runId": resolved}
        enriched.append(check)
    return enriched


def _resolve_run_id_via_check_run(self, check_run_id: int) -> int | None:
    """Resolve the workflow run id behind a check-run id (issue #2540).

    Two evidence sources, in cost order (see ``check_run_resolution``'s
    module docstring):

      1. ``_run_id_from_check_run_output``: an Actions job link the posting
         step embedded in the check run's caller-supplied ``output`` text --
         one API call, and the parse reuses the exact regex the direct-link
         path uses.
      2. ``_run_id_for_check_suite``: the payload's ``check_suite.id``
         matched against ``GET /actions/runs?check_suite_id=<id>`` -- for a
         check run the Actions app created from inside a workflow job, that
         suite is the workflow run's own.

    Returns ``None`` -- never guesses -- when both sources come up empty
    (fetch failure, external-CI check run, suite with no workflow run), so
    the caller's debouncing degrades to the pre-#2540
    definitive-on-first-failure behavior.
    """
    payload = self.gh.check_run(check_run_id)
    run_id = _run_id_from_check_run_output(payload)
    if run_id is not None:
        return run_id
    suite_id = _check_suite_id_from_payload(payload)
    if suite_id is None:
        return None
    runs = self.gh.workflow_runs_for_check_suite(suite_id)
    return _run_id_for_check_suite(runs, suite_id)
