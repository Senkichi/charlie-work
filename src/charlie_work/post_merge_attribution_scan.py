"""Post-merge outcome attribution scans (#1701, #1717).

The mechanical half of the two outcome-recording lanes: the experiment
report (#1701) folds the post-approval rework signals into
``post_approval_defect_rate``, but two outcome candidates were previously
only declared not-derivable -- a main-branch CI failure *after* a merge
was attributed to nothing, and a later fix/revert PR that names the
merged PR carried no recorded link. These functions record both, per
merged PR, in state.json's ``prs`` entries and as two new event kinds:

* ``post_merge_main_ci_failure`` -- a required check (or, when no
  ``auto_merge.required_checks`` are configured, any check) failed on
  the merged PR's merge-commit SHA. The probes are per-pass
  maintenance-lane work: check-run and workflow-run lookups against the
  SHA the merge produced, resolved from the PR's recorded ``merged_sha``
  (local lane) or the PR row's ``mergeCommitOid`` (``merged_pr_list``,
  REST). An entry is probed until its outcome is terminal, a
  probe-attempt cap runs out, or every run for the SHA is past its
  pending state -- so quiescence is cheap: terminal outcomes are stamped
  on the entry and never probed again.
* ``post_merge_followup_attribution`` -- an OPEN PR whose title/body
  matches a keyword-trigger reference (``reverts #123``-shaped) to a
  merged PR tracked in state. Recorded once per (merged PR, follow-up
  PR) pair; the matched text travels in the payload so the attribution
  is auditable against the PR it came from.

Event policy (per the ``_maybe_reclaim_superseded_main_ci`` template): no
durable write unless there is something worth one -- the overwhelmingly
common no-op pass emits nothing. Remote calls never happen while a state
lock is held: every probe takes its candidate snapshot under the lock,
releases it, works, and re-loads state when it records. Probe failures
are absorbed as values (``commit_check_runs`` and friends return ``None``
on failure), never raised past the caller; the pass-level exception
containment lives in the delegate lane that wires these in.

``app`` is the ``OrchestratorApp`` instance the delegate forwards; the
attributes reached are ``gh``, ``config``, ``paths.state_file``,
``host.clock.now``, ``_record_event`` and ``write_gate`` -- the same
surface the other per-pass lanes touch, so the suite can drive these
scans against a stub app directly.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf

# A merged PR is probed across passes until its outcome is terminal. The
# cap bounds the cost of the inconclusive cases (unreadable probes, a
# merge-commit SHA that fell out of the recent-merged REST window) at a
# fixed number of API round-trips per PR; an entry that exhausts it is
# left without an outcome rather than guessed at.
PROBE_ATTEMPT_CAP = 8

# Cap on the number of merged entries actually probed (remote calls made)
# per pass, so a burst of merges cannot turn one pass into a probe storm;
# everything still queued is picked up by the next pass.
PER_PASS_PROBE_CAP = 5

# Check-run conclusions that mean "this check did not fail" (a completed
# run whose conclusion is None is still pending, not exonerated).
_EXONERATED_CONCLUSIONS = frozenset({"success", "skipped", "neutral", "", None})

# Workflow-run conclusions that mean the Actions run itself failed.
# ``cancelled`` is deliberately excluded: on a merge-commit SHA a
# cancelled run is most often a superseded run being reaped or a queue
# re-write, and treating it as a defect would fire the event on healthy
# merges.
_FAILED_RUN_CONCLUSIONS = frozenset({"failure", "timed_out"})

# Pending states of the two run sources: a SHA whose runs have not all
# finished carries no usable outcome yet and is re-probed next pass.
_PENDING_STATES = frozenset({"queued", "in_progress"})

# Keyword-trigger reference scan over a follow-up PR's title/body:
# trigger word, a short span without another '#', then ``#<number>``.
# The closed trigger class (revert/fix/follow-up variants) plus the
# 120-char window keep irrelevant PR references down to the cases that
# plausibly name the prior merge.
_TRIGGER_REF_PATTERN = re.compile(
    r"\b(revert(?:s|ed)?|fix(?:es|ed)?|follow-?up)\b[^#]{0,120}?#(\d+)\b",
    re.IGNORECASE,
)

# Terminal stamp outcomes recorded on a merged entry's
# ``post_merge_ci_probe``. Anything else (``None`` with a reason) keeps
# the entry in the probe rotation until the attempt cap.
_CI_PROBE_TERMINAL_OUTCOMES = ("clean", "failure")

_FOLLOWUP_TAG = "post_merge_followups"


def probe_merged_main_ci(app: Any, *, state_file: Any, now: datetime | None = None) -> None:
    """Probe merged PRs for main-branch CI failure (part 1 of #1717).

    For every ``prs`` entry with ``status == "merged"`` and no terminal
    probe record: resolve the merge-commit SHA (the entry's own
    ``merged_sha`` when the merge went through the local gate, otherwise
    the REST ``mergeCommitOid`` from ``merged_pr_list``), then read the
    SHA's check runs and workflow runs. Outcomes:

    * any run still pending -- nothing recorded, no attempt spent; the
      entry is simply re-probed next pass (a queue that is still deep
      must not burn the attempt cap);
    * an unreadable probe pair, or an unresolved SHA -- inconclusive:
      the attempt counter advances so the cap above bounds the total
      cost per entry;
    * a required check (or any check, with no required list) failed --
      ``failure``: the ``post_merge_main_ci_failure`` event names what
      failed against the merged PR and its issue;
    * every finished run succeeded -- ``clean``.

    Remote calls happen outside the state lock; the record write
    re-loads state and merges the stamp into the entry's current
    content.
    """
    resolved_now = now if now is not None else app.host.clock.now()
    probed_at = resolved_now.astimezone(UTC).isoformat().replace("+00:00", "Z")
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        candidates: list[tuple[str, "dict[str, Any]", int]] = []
        for key, entry in state.get("prs", {}).items():
            if entry.get("status") != "merged":
                continue
            probe = entry.get("post_merge_ci_probe")
            if isinstance(probe, dict) and probe.get("outcome") in _CI_PROBE_TERMINAL_OUTCOMES:
                continue
            attempts = probe.get("attempts", 0) if isinstance(probe, dict) else 0
            if not isinstance(attempts, int) or attempts < 0:
                attempts = 0
            if attempts >= PROBE_ATTEMPT_CAP:
                continue
            number = entry.get("number")
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                continue
            candidates.append((key, entry, attempts))

    if not candidates:
        return
    log = logging.getLogger(__name__)
    rest_shas: "dict[int, str] | None" = None
    probed = 0
    for key, entry, attempts in candidates:
        if probed >= PER_PASS_PROBE_CAP:
            break
        number = int(entry["number"])
        sha = entry.get("merged_sha")
        if not (isinstance(sha, str) and sha):
            if rest_shas is None:
                rest_shas = _merge_commit_shas(app.gh)
            sha = rest_shas.get(number)
        if not (isinstance(sha, str) and sha):
            probed += 1
            _persist_probe_stamp(
                app,
                state_file,
                key,
                {"outcome": None, "reason": "sha_unresolved", "attempts": attempts + 1},
                probed_at=probed_at,
            )
            log.debug("post_merge_attribution: PR #%s has no merge-commit SHA yet", number)
            continue
        issue = entry.get("issue_number")
        checks = app.gh.commit_check_runs(sha)
        runs = app.gh.workflow_runs_for_head(sha)
        probed += 1
        if checks is None or runs is None:
            _persist_probe_stamp(
                app,
                state_file,
                key,
                {
                    "outcome": None,
                    "reason": "checks_unavailable",
                    "attempts": attempts + 1,
                    "merged_sha": sha,
                },
                probed_at=probed_at,
            )
            log.debug("post_merge_attribution: PR #%s probes unreadable", number)
            continue
        if _has_pending_runs(checks, runs):
            log.debug(
                "post_merge_attribution: PR #%s probes still running; re-probe next pass",
                number,
            )
            continue
        failures = sorted(set(_failed_check_names(checks)) | set(_failed_run_names(runs)))
        required = {str(name) for name in app.config.auto_merge.required_checks}
        # With a required list configured, only a failure in it is a
        # defect: an unrelated check failing is not the experiment's
        # outcome. Without one, any failure counts.
        defect_names = [name for name in failures if name in required] if required else failures
        stamp: "dict[str, Any]" = {
            "attempts": attempts + 1,
            "probed_at": probed_at,
            "merged_sha": sha,
            "observed_failures": failures,
        }
        if defect_names:
            stamp["outcome"] = "failure"
            stamp["failed_checks"] = defect_names
            _persist_probe_stamp(app, state_file, key, stamp, probed_at=probed_at)
            with _wf.state_lock(state_file):
                state = _wf.load_state(state_file)
                state = app._record_event(
                    state,
                    "post_merge_main_ci_failure",
                    {
                        "pr_number": number,
                        "issue_number": issue if isinstance(issue, int) and issue > 0 else None,
                        "merged_sha": sha,
                        "failed_checks": defect_names,
                        "observed_failures": failures,
                        "required_checks_applied": bool(required),
                    },
                )
                app.write_gate.save_state(state)
            log.info(
                "post_merge_attribution: merged PR #%s failed main-branch CI on %s",
                number,
                ", ".join(defect_names),
            )
        else:
            stamp["outcome"] = "clean"
            _persist_probe_stamp(app, state_file, key, stamp, probed_at=probed_at)


def scan_followup_pr_attribution(app: Any, *, state_file: Any) -> None:
    """Attribute follow-up fix/revert PRs to the merged PRs they name (part 2).

    Scans the open-PR list (``pr_list`` -- its GraphQL fields carry
    ``title`` and ``body``) for keyword-trigger ``#<number>`` references
    and links every reference to a PR recorded ``status == "merged"``
    in state to that merged PR. The pair cannot repeat: the merged
    entry's ``post_merge_followups`` list acts as the dedupe ledger, so
    the event is written exactly once per (merged, follow-up) pair, even
    when the follow-up's title and body both name the merge and even
    when the follow-up or the merged PR's row changes mid-scan (each
    record write re-loads state under the lock).
    """
    open_prs = app.gh.pr_list()
    if not isinstance(open_prs, list):
        return
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        merged_by_number: "dict[int, tuple[str, dict[str, Any]]]" = {}
        for key, entry in state.get("prs", {}).items():
            number = entry.get("number")
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                continue
            if entry.get("status") == "merged":
                merged_by_number[int(number)] = (key, entry)
        hits = _followup_hits(open_prs, merged_by_number)

    for merged_number, (key, followup, trigger, matched) in hits:
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            entry = state["prs"].get(key)
            if entry is None:
                # The merged row vanished mid-scan (state reset, prune);
                # there is nothing to attribute onto.
                continue
            already = entry.get(_FOLLOWUP_TAG)
            if not isinstance(already, list):
                already = []
            if followup in already:
                continue
            already.append(followup)
            entry[_FOLLOWUP_TAG] = already
            state["prs"][key] = entry
            state = app._record_event(
                state,
                "post_merge_followup_attribution",
                {
                    "pr_number": merged_number,
                    "issue_number": _stamped_issue(entry),
                    "followup_pr_number": followup,
                    "method": "keyword_mention",
                    "trigger": trigger,
                    "detail": matched,
                },
            )
            app.write_gate.save_state(state)


def _followup_hits(
    open_prs: "list[Any]",
    merged_by_number: "dict[int, tuple[str, dict[str, Any]]]",
) -> "list[tuple[int, tuple[str, int, str, str]]]":
    """(merged PR -> (state key, follow-up PR, trigger, matched text)) picks.

    The state key locates the dedupe ledger; two merged PRs named by one
    follow-up each get their own hit, and two open PRs naming the same
    merge each get their own hit.
    """
    hits: "list[tuple[int, tuple[str, int, str, str]]]" = []
    for pr in open_prs:
        if not isinstance(pr, dict):
            continue
        number = pr.get("number")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            continue
        title = pr.get("title") if isinstance(pr.get("title"), str) else ""
        body = pr.get("body") if isinstance(pr.get("body"), str) else ""
        text = "\n".join(part for part in (title, body) if part)
        if not text:
            continue
        for match in _TRIGGER_REF_PATTERN.finditer(text):
            referenced = int(match.group(2))
            entry = merged_by_number.get(referenced)
            if entry is None:
                continue
            hits.append(
                (
                    referenced,
                    (entry[0], int(number), match.group(1).lower(), match.group(0)[:200]),
                )
            )
    return hits


def _persist_probe_stamp(
    app: Any, state_file: Any, pr_key: str, stamp: "dict[str, Any]", *, probed_at: str
) -> None:
    """Merge ``stamp`` into a merged entry's ``post_merge_ci_probe`` record.

    Called after the probe's remote calls, so the write re-loads state
    under the lock: anything the entry gained while the probing ran
    stays, and this stamp only adds its own keys.
    """
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        entry = state["prs"].get(pr_key)
        if entry is None:
            # The row vanished between the candidate snapshot and this
            # write -- nothing to stamp, nothing to lose.
            return
        entry["post_merge_ci_probe"] = stamp | {"probed_at": probed_at}
        state["prs"][pr_key] = entry
        app.write_gate.save_state(state)


def _stamped_issue(entry: "dict[str, Any]") -> "int | None":
    """The stamped issue of a merged entry; None when absent/None-valued."""
    issue = entry.get("issue_number")
    if isinstance(issue, int) and issue > 0:
        return issue
    return None


def _merge_commit_shas(gh: Any) -> "dict[int, str]":
    """PR number -> merge-commit SHA for the recently merged REST window.

    ``_normalize_rest_pr`` maps the REST row's ``merge_commit_sha`` onto
    ``mergeCommitOid``; the GraphQL PR list only covers PRs, so this
    merged REST list is the per-pass bulk source for other PRs' SHAs.
    """
    rows = gh.merged_pr_list()
    if not isinstance(rows, list):
        return {}
    shas: "dict[int, str]" = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        number = row.get("number")
        sha = row.get("mergeCommitOid")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            continue
        if not (isinstance(sha, str) and sha):
            continue
        shas[int(number)] = sha
    return shas


def _failed_check_names(checks: Any) -> "list[str]":
    """Human names of failed completed check runs in a raw REST payload."""
    if not isinstance(checks, list):
        return []
    names: "list[str]" = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        # Anything whose completed conclusion is not in the exonerated
        # set is a failure (a completed run with a None conclusion reads
        # as pending, handled separately by the pending check).
        if (
            check.get("status") != "completed"
            or check.get("conclusion") in _EXONERATED_CONCLUSIONS
        ):
            continue
        name = check.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _failed_run_names(run_rows: Any) -> "list[str]":
    """Human names of failed completed Actions runs in a raw REST payload."""
    if not isinstance(run_rows, list):
        return []
    names: "list[str]" = []
    for run in run_rows:
        if not isinstance(run, dict):
            continue
        if (
            run.get("status") != "completed"
            or run.get("conclusion") not in _FAILED_RUN_CONCLUSIONS
        ):
            continue
        name = run.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _has_pending_runs(checks: Any, run_rows: Any) -> bool:
    """True while any check run or Actions run for the SHA is still pending."""
    for payload in (checks, run_rows):
        if not isinstance(payload, list):
            continue
        for run in payload:
            if not isinstance(run, dict):
                continue
            if run.get("status") in _PENDING_STATES:
                return True
            # A ``completed`` run whose conclusion is absent never
            # resolves its result in the REST payloads; treat it as
            # still open rather than exonerated.
            if run.get("status") == "completed" and run.get("conclusion") is None:
                return True
    return False
