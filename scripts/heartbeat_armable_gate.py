"""``check_armable_backlog`` and its dependency/sweep gates for ``heartbeat_check.py``.

The armable-backlog check ("is the armed runway thin while un-triaged,
armable issues sit idle?") lives here together with the gates that keep
deliberately-parked issues out of its un-triaged pool (issue #2004): the
body-blocker gate (the orchestrator's own ``parse_blockers``) and the
config-retirement sweep gate (the ``DEPRECATED_CONFIG_KEYS`` registry's
``removal_issue`` numbers, anchored on this script's own repo).

Loaded from ``heartbeat_check.py`` via ``importlib`` from the sibling script
path, never a bare ``import`` -- matching how #1895 loads
``heartbeat_event_alarms.py``, #1879 loads ``heartbeat_local_repo.py``, and
``git_push_lint_hook.py`` loads ``worker_stop_gate.py``: ``scripts/`` is not
a package and is deliberately kept off ``sys.path`` by the test harness
(``tests/_script_loader.py``). ``heartbeat_check`` re-exports
``check_armable_backlog`` and the ``ARMABLE_*``/``ARMED_LABEL`` constants,
so ``hb.check_armable_backlog`` / ``hb.ARMABLE_GATING_LABELS`` attribute
references in tests keep resolving unchanged. This module is never run
standalone.

Extracted out of ``heartbeat_check.py`` to create file-size ratchet headroom
(``file_size_ratchet_baseline/scripts/heartbeat_check.py.count``, PR #2023
rework) -- no behavior change.

Stdlib-only, same constraint as ``heartbeat_check.py`` itself
(``scripts/README.md``): no ``charlie_work`` or third-party imports beyond
the two guarded leaf imports below, which moved here with their sole
consumer -- the same pattern that moved ``EXPECTED_OPERATIONAL_KINDS``'s
guarded ``charlie_work.event_kinds`` import into ``heartbeat_event_alarms``
with ``check_warning_events``. Never an import back into
``heartbeat_check`` -- that would cycle through its loader block, which is
also why ``Report``/``RepoInfo`` are ``TYPE_CHECKING``-only names.

Unlike the other siblings this module does need ``heartbeat_check``
functions at call time (``run_gh_json``, ``get_dispatch_cap``) -- helpers
tests monkeypatch on the loaded ``hb`` module, so mirroring them here would
silently disconnect those patches. ``heartbeat_check``'s loader block calls
``bind(sys.modules[__name__])`` after ``exec_module`` (``__main__`` on a
script run, ``heartbeat_check`` under the test loader -- registered in
``sys.modules`` before exec in both cases), and this module reads the
helpers off that handle at call time, so ``hb.run_gh_json`` /
``hb.get_dispatch_cap`` monkeypatches keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # ``heartbeat_check`` is not importable as a module at runtime
    # (``scripts/`` is not a package and stays off ``sys.path``); these names
    # exist so the moved function keeps its original annotations
    # byte-identically. A runtime import would cycle through
    # ``heartbeat_check``'s own loader block.
    from heartbeat_check import RepoInfo, Report

# armable-backlog (2026-08-23): "plenty to work on" = one full wave of armed,
# unclaimed issues (dispatch.max_concurrent_sessions); fallback when the cap
# is unreadable. Gating labels mark an open issue as *triaged but deliberately
# not armed*, so it leaves the un-triaged "armable" pool; keep this set in step
# with the label taxonomy both repos share (`needs-design`, `human-action`,
# `blocked`) plus GitHub's default terminal labels.
ARMABLE_RUNWAY_FLOOR_DEFAULT = 3
ARMED_LABEL = "automated-ready"
ARMABLE_GATING_LABELS: frozenset[str] = frozenset(
    {"blocked", "needs-design", "human-action", "question", "wontfix", "duplicate", "invalid"}
)
ARMABLE_PREVIEW_LIMIT = 8

# Issue #2004: the armable pool's dependency gate reuses the orchestrator's own
# body parser (single point of enforcement), and its sweep gate reuses the
# deprecated-key registry itself -- config_deprecations.py is deliberately the
# single registry with no second marker, so a `removal_issue` number IS the
# marker. Both sit behind guarded imports; when one is unavailable its gate
# degrades to a no-op and check_armable_backlog's emitted verdict line carries
# a caveat naming the degradation.
try:
    from charlie_work.github_body_scan import parse_blockers
except ImportError:
    parse_blockers = None

try:
    from charlie_work.config_deprecations import DEPRECATED_CONFIG_KEYS
except ImportError:
    DEPRECATED_CONFIG_KEYS = None

# The registry's ``removal_issue`` numbers are issues in the orchestrator's own
# repo -- the checkout this script runs from (this module lives at
# <repo>/scripts/, same as heartbeat_check.py). A same-numbered issue in a
# sibling managed repo is a different issue entirely, so the sweep gate is
# anchored on this root, not applied fleet-wide.
_SELF_REPO_ROOT = Path(__file__).resolve().parent.parent

# The loading ``heartbeat_check`` module object, bound by its loader block via
# ``bind`` (see module docstring). Read through it -- never captured as locals
# -- so call-time attribute lookups see test monkeypatches on ``hb``.
_hb: Any = None


def bind(heartbeat_check_module: Any) -> None:
    """Bind the module handle this file's check reads ``hb`` helpers through."""
    global _hb
    _hb = heartbeat_check_module


def _has_open_body_blocker(issue: dict[str, Any], open_numbers: set[int]) -> bool:
    """True when the issue body declares a blocker that is still open.

    Uses the orchestrator's own ``parse_blockers``; "open" is membership in the
    open-issue list already fetched (a closed blocker is absent from it).
    """
    if parse_blockers is None:
        return False
    return any(n in open_numbers for n in parse_blockers(issue.get("body") or ""))


def check_armable_backlog(
    report: Report,
    repo: RepoInfo,
    blocked_numbers: set[int] | None,
    blocked_err: str,
) -> None:
    """Is the armed runway thin while un-triaged, armable issues sit idle?

    ``dispatch-coverage`` asks "did the fleet pick up what is armed?"; this
    check asks the question upstream of it: "is there enough armed work for
    the fleet to pick up, and if not, is that because the backlog is
    genuinely empty or because nobody has triaged it?" (2026-08-23: both
    lanes were about to idle with 12 + 39 open issues carrying no label at
    all -- neither ``automated-ready`` nor any gating label -- so the fleet
    starved with work available.)

    Three buckets over the open issues:

    * ``runway``  -- ``automated-ready``, no ``agent:*`` label, not blocked:
      what dispatch can take next. Healthy when ``>= floor``.
    * ``active``  -- carries an ``agent:*`` label (in flight / terminal).
    * ``armable`` -- no ``agent:*`` label, not ``automated-ready``, and no
      *gating* label (``ARMABLE_GATING_LABELS``), blocked-by-dependency
      entry, open body-declared blocker, or ``removal_issue`` membership in
      ``DEPRECATED_CONFIG_KEYS`` (the config-retirement sweep files those
      issues unarmed and arms them itself -- issue #2004). This is the
      un-triaged pool: every issue here is either a missed arm or a missed
      gate, and a triage pass drives it to zero.

    Verdict:

    * runway ``>= floor``                      -> OK (plenty to work on)
    * runway ``< floor`` and armable is empty  -> OK (genuinely empty)
    * runway ``< floor`` and armable non-empty -> ANOMALY: triage needed

    ``floor`` is the repo's ``dispatch.max_concurrent_sessions`` cap (one
    full wave of work), falling back to ``ARMABLE_RUNWAY_FLOOR_DEFAULT``.

    Degraded blocked-issue lookup (``blocked_err``) can only *inflate* both
    ``runway`` and ``armable``: an inflated runway can turn an anomaly into
    a false OK, an inflated armable can turn an OK into a false anomaly. The
    caveat is surfaced on whichever verdict is emitted rather than guessed
    around.

    The two guarded imports degrade the same direction: a missing
    ``parse_blockers`` or registry leaves body-declared blockers and
    sweep-owned removal issues inside ``armable`` -- a possible false
    anomaly, never a hidden one -- and each degradation is named in a
    caveat on the emitted verdict line.
    """
    check = f"armable-backlog {repo.slug}"
    if repo.local_issues_enabled:
        report.ok(check, _hb.LOCAL_ONLY_SKIP_DETAIL)
        return
    caveats: list[str] = []
    if blocked_err:
        caveats.append(f"blocked-issue lookup degraded: {blocked_err}")
    if parse_blockers is None:
        caveats.append("body-blocker gate degraded: charlie_work.github_body_scan unavailable")
    if DEPRECATED_CONFIG_KEYS is None and repo.repo_root.resolve() == _SELF_REPO_ROOT:
        caveats.append(
            "sweep-registry gate degraded: charlie_work.config_deprecations unavailable"
        )
    caveat = f" ({'; '.join(caveats)})" if caveats else ""
    args = [
        "issue",
        "list",
        "-R",
        repo.slug,
        "--state",
        "open",
        "--json",
        "number,labels,body",
        "--limit",
        str(_hb.ISSUE_LIST_LIMIT),
    ]
    ok, data, err = _hb.run_gh_json(args, repo.repo_root)
    if not ok:
        report.anom(check, f"{err}{caveat}")
        return

    open_numbers = {issue["number"] for issue in data}
    # The retirement sweep owns its removal issues end to end: it files them
    # unarmed ("do not label by hand") and marks them `automated-ready` itself
    # once each key's quiet window elapses (config_retirement_sweep). The
    # registry is the marker -- there is deliberately no second label
    # (config_deprecations.py) -- and the gate only applies to this script's
    # own repo, where those issue numbers live.
    sweep_owned = (
        {entry.removal_issue for entry in DEPRECATED_CONFIG_KEYS}
        if DEPRECATED_CONFIG_KEYS is not None and repo.repo_root.resolve() == _SELF_REPO_ROOT
        else frozenset()
    )
    runway: list[int] = []
    active = 0
    gated = 0
    armable: list[int] = []
    for issue in data:
        number = issue["number"]
        names = {label["name"] for label in issue.get("labels", [])}
        if any(n.startswith("agent:") for n in names):
            active += 1
            continue
        is_blocked = blocked_numbers is not None and number in blocked_numbers
        if is_blocked or names & ARMABLE_GATING_LABELS:
            gated += 1
            continue
        if ARMED_LABEL in names:
            runway.append(number)
            continue
        # Un-armed candidate: `fleet status` only evaluates blockers for
        # `automated-ready` issues, so an unlabelled issue never reaches
        # blocked_numbers -- evaluate its body here; a sweep-owned removal
        # issue is likewise parked rather than un-triaged (issue #2004).
        if number in sweep_owned or _has_open_body_blocker(issue, open_numbers):
            gated += 1
            continue
        armable.append(number)

    cap = _hb.get_dispatch_cap(repo.config_path) if repo.config_path else None
    floor = cap if cap is not None else ARMABLE_RUNWAY_FLOOR_DEFAULT
    facts = (
        f"runway={len(runway)} floor={floor} armable={len(armable)} "
        f"active={active} gated={gated} open={len(data)}"
    )

    if len(runway) >= floor:
        report.ok(check, f"plenty armed; {facts}{caveat}")
    elif not armable:
        report.ok(
            check, f"runway thin but backlog genuinely empty of armable issues; {facts}{caveat}"
        )
    else:
        preview = sorted(armable)[:ARMABLE_PREVIEW_LIMIT]
        more = len(armable) - len(preview)
        suffix = f" (+{more} more)" if more > 0 else ""
        report.anom(
            check,
            f"runway thin ({len(runway)} < floor {floor}) while {len(armable)} "
            f"un-triaged armable issue(s) sit idle: {preview}{suffix} -- triage: "
            f"label each `{ARMED_LABEL}` or one of {sorted(ARMABLE_GATING_LABELS)}"
            f"; {facts}{caveat}",
        )
