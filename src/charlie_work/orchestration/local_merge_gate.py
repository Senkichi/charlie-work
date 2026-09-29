"""Asynchronous local merge-gate delegates for ``OrchestratorApp`` (issue #1974).

Origin: the #1844 gate ran ``run_full_suite`` synchronously inside the
supervisor pass and looped over *every* approved record -- a 33-47 minute
suite per record froze the whole fleet for ``N_approved x suite_duration``.
That violates the CLAUDE.md "adapters must not block on worker completion"
invariant: anything that can run for minutes launches with ``Popen`` and is
observed on later passes, exactly like worker sessions.

The gate is now a bounded two-phase state machine per approved record:

* **launch pass** -- the base-sync merge runs inline (fast git plumbing),
  then :func:`charlie_work.local_suite_runner.launch_suite_gate` spawns
  ``python -m charlie_work.local_suite_runner`` with ``Popen`` (non-blocking,
  stdout+stderr streamed to ``suite.log`` under the record's dispatch dir).
  The claim -- ``local_suite_pid``, ``local_suite_process_start_time``,
  ``local_suite_started_at``, ``local_suite_log``, ``local_suite_gate_dir``,
  ``local_suite_head``, ``local_suite_base_sha``, ``local_suite_argv`` --
  persists on the lane record and the pass returns ``suite_launched``.
* **later passes** -- ``suite-result.json`` settles the gate: green checks
  that neither the branch head nor the base moved under the suite (drift ->
  re-sync + relaunch, never a merge of an untested pairing), then merges;
  red routes to rework through the existing #1972 capped path. Runtime past
  ``SUITE_TIMEOUT_SECONDS`` kills the process tree and routes as a suite
  failure. A dead runner with no result is *never* green -- the gate
  relaunches bounded by ``LOCAL_SUITE_GATE_MAX_ORPHANS``, then escalates the
  infrastructure anomaly.

**Crash safety.** The wrapper writes ``suite-runner.json`` (pid + start-time
fingerprint + claim fields) before it starts the suite, so a supervisor that
dies between the ``Popen`` and the claim write re-attaches the live runner
via the pid file instead of launching a second one, and a supervisor that
dies mid-suite resumes from the persisted claim + result file. A result file
is accepted only when its ``head_sha`` matches the head under test -- a
stale artifact is indistinguishable from a missing one, and a missing one is
never green.

**One gate per repo.** Base merges serialize anyway, so at most one suite is
in flight: other approved records get a ``local_merge_deferred`` event and
wait for the next pass. The in-flight flag is seeded from the persisted
claims and live runner pid files before the ordered walk
(``_local_gate_any_in_flight``), so the bound does not depend on where the
record holding the gate sorts.

Every top-level ``def`` is installed on ``OrchestratorApp`` by
``workflow_delegation._install_delegates``; workflow helpers are reached
through ``_wf.<name>`` (the module-object form the monkeypatch seams need).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work import local_suite_runner
from charlie_work.labels import TransitionOutcome
from charlie_work.local_lane import (
    SUITE_TIMEOUT_SECONDS,
    _iso_dt,
    branch_diff,
    branch_head_sha,
    ensure_branch_worktree,
    local_base_branch,
    local_pr_records,
    merge_branch_into_base,
    resolve_ref_sha,
    suite_command_argv,
    worktree_for_branch,
)
from charlie_work.process_utils import is_pid_alive
from charlie_work.safe_path import contains
from charlie_work.state import without_review_dispatch_claim
from charlie_work.worktree import (
    _merge_update_rework_branch,
    remove_review_checkout,
    remove_worktree,
)

# In-flight claim fields on the lane record. Clearing writes None to each --
# a cleared field is indistinguishable from an unlaunched one, which is what
# lets the launch path treat every resolved/crashed gate uniformly.
LOCAL_SUITE_CLAIM_FIELDS = (
    "local_suite_pid",
    "local_suite_process_start_time",
    "local_suite_started_at",
    "local_suite_log",
    "local_suite_gate_dir",
    "local_suite_head",
    "local_suite_base_sha",
    "local_suite_argv",
)

# Bounds on the two self-healing relaunch paths. ``resync`` covers a base (or
# head) that keeps moving under the suite; ``orphan`` covers a runner that
# died without writing a result. Both are per-gate-episode counters: the
# launch write sets them explicitly, and only a terminal resolution (merge /
# rework / escalation) ends the episode. Unbounded relaunches would silently
# burn suite-hours on a repo that can never converge -- at the bound the gate
# escalates the infrastructure anomaly like any other local_merge_error.
LOCAL_SUITE_GATE_MAX_RESYNCS = 3
LOCAL_SUITE_GATE_MAX_ORPHANS = 3


def _local_merge_approved(self) -> list[dict[str, Any]]:
    """Merge approved local branches: base-sync, async full suite, advance base.

    For every lane record whose recorded verdict is ``approved``, in issue
    order:

    1. A record with a live gate claim is *resolved*: its result file /
     liveness / age are evaluated and the suite is never waited on
     (``_local_gate_poll``).
    2. An approved record with no claim and a live ``suite-runner.json`` pid
     file re-attaches the crashed write's runner (``_local_gate_adopt``).
    3. The first remaining approved record launches: the base-sync merge
     runs inline, then the suite is spawned detached
     (``_local_gate_launch``). At most one gate is in flight per repo --
     later approved records defer to the next pass.
    4. Green results re-verify base/head stability, then advance the base
     (``_local_gate_finalize_merge`` -- fast-forward or ``--no-ff``, the
     same terminal bookkeeping as before).
    """
    results: list[dict[str, Any]] = []
    state = _wf.load_state_locked(self.paths.state_file)
    records = sorted(local_pr_records(state).items(), key=lambda kv: int(kv[0]))
    # The one-gate bound cannot be derived from iteration order: a flag that
    # fills in only as the walk proceeds never sees a claimed higher-numbered
    # record before an unclaimed lower-numbered one, and the lower record
    # would launch a second concurrent suite beside the first. Seed from the
    # persisted claims and live runner pid files so the in-flight gate holds
    # for the whole pass, wherever its record sorts.
    gate_in_flight = self._local_gate_any_in_flight(records)

    for pr_key, record in records:
        pr_number = int(pr_key)
        issue_number = int(record.get("issue_number") or pr_number)
        entry: dict[str, Any] = {"issue": issue_number, "pr": pr_number}
        branch = str(record.get("branch") or record.get("headRefName") or "")
        base_ref = str(record.get("baseRefName") or local_base_branch(self.repo_root) or "HEAD")
        claimed = bool(record.get("local_suite_pid"))

        if record.get("status") != "approved":
            if claimed:
                # The record moved on (operator escalation, packet void)
                # while its suite ran. The tested head is superseded -- the
                # stray suite must die and its claim must not resolve later.
                self._local_gate_abort(
                    pr_key,
                    record,
                    branch=branch,
                    reason="record left approved with a suite in flight",
                )
                entry["outcome"] = "gate_aborted"
                results.append(entry)
            continue

        decision = self._review_decision(pr_number)
        reviewed_head = decision.get("reviewed_head_sha")
        live_head = branch_head_sha(self.repo_root, branch) if branch else None

        # The merge gate only ever merges the exact head the reviewer
        # approved -- a verdict pinned to a superseded head is voided by the
        # packet builder's stale-verdict reset, but double-check here anyway
        # so a torn state can never authorize the wrong head.
        if live_head is None:
            if claimed:
                self._local_gate_abort(
                    pr_key,
                    record,
                    branch=branch,
                    reason=f"branch {branch!r} does not resolve",
                )
            entry["outcome"] = "error"
            entry["detail"] = f"branch {branch!r} does not resolve"
            results.append(entry)
            continue
        # A claimed record's expected head is ``local_suite_head`` -- the
        # post-sync-merge sha the suite is actually testing -- not
        # ``reviewed_head``: the gate's own base-sync merge at launch
        # legitimately advances the branch tip past the reviewed sha.
        expected_head = record.get("local_suite_head") if claimed else reviewed_head
        if expected_head and expected_head != live_head:
            # Head moved after approval. The common cause is this gate's own
            # base-sync merge at launch -- the reviewed delta is unchanged,
            # only the head moved. Honour the approval when the live diff
            # still matches the recorded patch-id; any genuinely new content
            # (a rogue push, rework landing mid-gate) falls through to the
            # packet phase's rebuild + re-review -- and kills the suite,
            # which is testing a superseded head.
            reviewed_patch = decision.get("reviewed_patch_id")
            live_diff = branch_diff(self.repo_root, base_ref, branch)
            if not (
                reviewed_patch
                and live_diff is not None
                and _wf._calculate_patch_id(live_diff) == reviewed_patch
            ):
                if claimed:
                    self._local_gate_abort(
                        pr_key,
                        record,
                        branch=branch,
                        reason="branch head moved under the gate",
                    )
                entry["outcome"] = "skipped_head_moved"
                results.append(entry)
                continue

        if claimed:
            if self._local_gate_poll(
                pr_key=pr_key,
                record=record,
                entry=entry,
                branch=branch,
                base_ref=base_ref,
                live_head=live_head,
                decision=decision,
            ):
                gate_in_flight = True
            results.append(entry)
            continue

        if self._local_gate_adopt(pr_key=pr_key, record=record, entry=entry):
            gate_in_flight = True
            results.append(entry)
            continue

        if gate_in_flight:
            entry["outcome"] = "deferred"
            entry["detail"] = "another local merge gate is in flight"
            self._local_gate_event(
                "local_merge_deferred",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "detail": "another local merge gate is in flight",
                },
            )
            results.append(entry)
            continue

        gate_in_flight = self._local_gate_launch(
            pr_key=pr_key,
            record=record,
            entry=entry,
            branch=branch,
            base_ref=base_ref,
            decision=decision,
            reason="new",
        )
        results.append(entry)
    return results


def _local_gate_any_in_flight(
    self,
    records: list[tuple[str, dict[str, Any]]],
) -> bool:
    """Whether any lane record already has a merge-gate suite in flight.

    Seeds ``gate_in_flight`` before ``_local_merge_approved``'s ordered walk.
    A record holds the gate when it carries a persisted claim
    (``local_suite_pid`` -- a dead claimed pid still holds it, because the
    orphan path relaunches rather than freeing the gate mid-pass) or when it
    is still ``approved`` with a live ``suite-runner.json`` the adopt step
    will re-attach. Claims count regardless of record status: a claimed
    record leaving ``approved`` is aborted later in the same pass, and
    deferring a launch one pass beats two suites running side by side.
    """
    for pr_key, record in records:
        if record.get("local_suite_pid"):
            return True
        if record.get("status") != "approved":
            continue
        paths = local_suite_runner.suite_gate_paths(self.paths.dispatches, int(pr_key))
        if self._local_gate_live_runner_meta(paths) is not None:
            return True
    return False


def _local_gate_event(
    self,
    kind: str,
    payload: dict[str, Any],
    *,
    level: str | None = None,
) -> None:
    """Emit one gate event through the standard locked state write."""
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state = self._record_event(  # event-consumer: audit-only -- forwarding wrapper; the literal kind is chosen at each self._local_gate_event(...) call site, which is itself scanned (same shape as WriteGate.append_event)
            state, kind, payload, level=level
        )
        self.write_gate.save_state(state)


def _local_gate_update(
    self,
    pr_key: str,
    updates: dict[str, Any],
    *,
    event: tuple[str, dict[str, Any]] | None = None,
) -> None:
    """Merge ``updates`` into the lane record; optionally emit one event.

    Single locked write for both so a claim change and its audit event can
    never be torn apart by a crash between two saves.
    """
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(pr_key, {})
        state["prs"][pr_key] = {**pr_state, **updates}
        if event is not None:
            kind, payload = event
            state = self._record_event(  # event-consumer: audit-only -- forwarding wrapper; the literal kind arrives in the (kind, payload) tuple built at each self._local_gate_update(...) call site, which is itself scanned
                state, kind, payload
            )
        self.write_gate.save_state(state)


def _local_gate_clear_claim(self, pr_key: str) -> None:
    """Erase the in-flight claim; counters survive for the bound checks."""
    self._local_gate_update(pr_key, {field: None for field in LOCAL_SUITE_CLAIM_FIELDS})


def _local_gate_abort(
    self,
    pr_key: str,
    record: dict[str, Any],
    *,
    branch: str,
    reason: str,
) -> None:
    """Kill a claimed suite whose record can no longer use its result."""
    pid = record.get("local_suite_pid")
    if isinstance(pid, int) and pid > 0:
        self.write_gate.kill_process_tree(pid, record.get("local_suite_process_start_time"))
    self._local_gate_clear_claim(pr_key)
    self._local_gate_event(
        "local_merge_deferred",
        {
            "pr_number": int(pr_key),
            "issue_number": record.get("issue_number"),
            "detail": f"in-flight merge gate aborted: {reason}",
        },
    )


def _local_gate_live_runner_meta(
    self,
    paths: local_suite_runner.SuiteGatePaths,
) -> dict[str, Any] | None:
    """The wrapper's pid-file metadata when it names a live runner process.

    One liveness predicate shared by ``_local_gate_adopt`` and
    ``_local_gate_any_in_flight``: a pid file only holds the gate while the
    fingerprint-checked process it names is still alive. A dead or malformed
    file means nothing is in flight -- the launch path scrubs the stale
    artifacts anyway.
    """
    meta = local_suite_runner.read_gate_pid(paths)
    if meta is None:
        return None
    pid = meta.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    if not is_pid_alive(pid, meta.get("process_start_time")):
        return None
    return meta


def _local_gate_adopt(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    """Re-attach a live runner whose claim write never landed (crash window).

    Returns True when a pid file names a live, fingerprint-matched runner --
    the claim is rebuilt from the file and the record keeps waiting instead
    of double-launching. A dead or malformed pid file is ignored: the launch
    path scrubs stale artifacts anyway.
    """
    pr_number = int(pr_key)
    paths = local_suite_runner.suite_gate_paths(self.paths.dispatches, pr_number)
    meta = self._local_gate_live_runner_meta(paths)
    if meta is None:
        return False
    pid = meta["pid"]
    self._local_gate_update(
        pr_key,
        {
            "local_suite_pid": pid,
            "local_suite_process_start_time": meta.get("process_start_time"),
            "local_suite_started_at": meta.get("started_at"),
            "local_suite_log": str(paths.log),
            "local_suite_gate_dir": str(paths.gate_dir),
            "local_suite_head": meta.get("head_sha"),
            "local_suite_base_sha": meta.get("base_sha"),
            "local_suite_argv": meta.get("suite_argv"),
            "local_suite_resync_count": int(record.get("local_suite_resync_count") or 0),
            "local_suite_orphan_count": int(record.get("local_suite_orphan_count") or 0),
        },
    )
    entry["outcome"] = "suite_running"
    entry["pid"] = pid
    entry["adopted"] = True
    return True


def _local_gate_poll(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    live_head: str,
    decision: dict[str, Any],
) -> bool:
    """Resolve a claimed gate without ever blocking on the suite.

    Returns True while a suite remains in flight for this record (still
    running, or just relaunched); False once the gate resolved to a terminal
    outcome this pass.
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    paths = local_suite_runner.suite_gate_paths(self.paths.dispatches, pr_number)
    gate_head = record.get("local_suite_head")

    result = local_suite_runner.read_gate_result(paths)
    if result is not None and result.get("head_sha") != gate_head:
        # Artifact of an earlier gate epoch -- never resolves this claim.
        result = None
    if result is not None:
        return self._local_gate_resolve_result(
            pr_key=pr_key,
            record=record,
            entry=entry,
            branch=branch,
            base_ref=base_ref,
            live_head=live_head,
            decision=decision,
            result=result,
            paths=paths,
        )

    pid = record.get("local_suite_pid")
    start_time = record.get("local_suite_process_start_time")
    if isinstance(pid, int) and pid > 0:
        if is_pid_alive(pid, start_time):
            started = _iso_dt(record.get("local_suite_started_at"))
            age = (datetime.now(UTC) - started).total_seconds() if started else 0
            if age > SUITE_TIMEOUT_SECONDS:
                killed = self.write_gate.kill_process_tree(pid, start_time)
                self._local_gate_clear_claim(pr_key)
                entry["outcome"] = "suite_failed"
                entry["timed_out"] = True
                self._local_gate_event(
                    "local_suite_result",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "ok": False,
                        "timed_out": True,
                        "pid": pid,
                        "killed_pids": killed,
                        "argv": record.get("local_suite_argv") or [],
                        "head_sha": gate_head,
                        "tail": local_suite_runner.read_log_tail(paths.log),
                    },
                )
                entry["routed_to"] = self._local_route_merge_rework(
                    pr_number,
                    issue_number,
                    record,
                    decision,
                    reason="suite_failed",
                    note=(
                        "The full test suite exceeded the merge gate's "
                        f"{SUITE_TIMEOUT_SECONDS}s limit and was killed. The "
                        "code changes are already approved; make the suite "
                        "pass within the limit. Tail of the suite output:\n\n"
                        f"```\n{local_suite_runner.read_log_tail(paths.log)}\n```"
                    ),
                )
                return False
            entry["outcome"] = "suite_running"
            entry["pid"] = pid
            return True

    # The recorded pid is dead or its fingerprint no longer verifies. Before
    # declaring an orphan, consult the pid file the wrapper wrote for itself:
    # a live, fingerprint-matched pid there means the runner is alive under a
    # corrected identity -- heal the claim rather than double-launching a
    # second suite next to the first.
    meta = local_suite_runner.read_gate_pid(paths)
    meta_pid = meta.get("pid") if meta else None
    if (
        isinstance(meta_pid, int)
        and meta_pid > 0
        and meta is not None
        and meta.get("head_sha") == gate_head
        and is_pid_alive(meta_pid, meta.get("process_start_time"))
    ):
        self._local_gate_update(
            pr_key,
            {
                "local_suite_pid": meta_pid,
                "local_suite_process_start_time": meta.get("process_start_time"),
                "local_suite_started_at": meta.get("started_at")
                or record.get("local_suite_started_at"),
            },
        )
        entry["outcome"] = "suite_running"
        entry["pid"] = meta_pid
        return True

    # Dead pid + no usable result: the wrapper exited without reporting --
    # never green. Relaunch bounded, then escalate the anomaly.
    orphans = int(record.get("local_suite_orphan_count") or 0) + 1
    self._local_gate_clear_claim(pr_key)
    if orphans > LOCAL_SUITE_GATE_MAX_ORPHANS:
        detail = (
            f"merge-gate suite runner died without writing a result "
            f"{orphans - 1} times for pr-{pr_number}"
        )
        entry["outcome"] = "error"
        entry["detail"] = detail
        self._local_merge_error_escalate(pr_number, issue_number, branch, detail)
        return False
    return self._local_gate_launch(
        pr_key=pr_key,
        record=record,
        entry=entry,
        branch=branch,
        base_ref=base_ref,
        decision=decision,
        reason="orphaned_gate",
        orphan_count=orphans,
    )


def _local_gate_resolve_result(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    live_head: str,
    decision: dict[str, Any],
    result: dict[str, Any],
    paths: local_suite_runner.SuiteGatePaths,
) -> bool:
    """A head-matched result exists: drift-check, then merge or rework."""
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    gate_head = record.get("local_suite_head")
    gate_base = record.get("local_suite_base_sha")
    base_now = resolve_ref_sha(self.repo_root, base_ref)
    drifted = live_head != gate_head or (
        gate_base is not None and base_now is not None and base_now != gate_base
    )
    if drifted:
        # The tested pairing is stale -- never merge a head the suite did
        # not see against the current base (issue #1974 AC).
        resyncs = int(record.get("local_suite_resync_count") or 0) + 1
        self._local_gate_clear_claim(pr_key)
        if resyncs > LOCAL_SUITE_GATE_MAX_RESYNCS:
            detail = (
                f"base moved under the in-flight merge gate {resyncs - 1} "
                f"times for pr-{pr_number}; cannot converge on a tested pairing"
            )
            entry["outcome"] = "error"
            entry["detail"] = detail
            self._local_merge_error_escalate(pr_number, issue_number, branch, detail)
            return False
        return self._local_gate_launch(
            pr_key=pr_key,
            record=record,
            entry=entry,
            branch=branch,
            base_ref=base_ref,
            decision=decision,
            reason="resync",
            resync_count=resyncs,
        )

    ok = bool(result.get("ok"))
    argv = record.get("local_suite_argv") or result.get("suite_argv") or []
    tail = local_suite_runner.read_log_tail(paths.log)
    self._local_gate_update(
        pr_key,
        {field: None for field in LOCAL_SUITE_CLAIM_FIELDS},
        event=(
            "local_suite_result",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "ok": ok,
                "argv": argv,
                "returncode": result.get("returncode"),
                "head_sha": gate_head,
                "tail": _wf._truncate_for_event(tail),
                "duration_seconds": result.get("duration_seconds"),
            },
        ),
    )
    if ok:
        self._local_gate_event(
            "local_suite_ok",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "argv": argv,
                "returncode": result.get("returncode"),
                "head_sha": gate_head,
                "duration_seconds": result.get("duration_seconds"),
            },
        )
        self._local_gate_finalize_merge(
            pr_key=pr_key,
            record=record,
            entry=entry,
            branch=branch,
            base_ref=base_ref,
            live_head=live_head,
            decision=decision,
        )
        return False

    entry["outcome"] = "suite_failed"
    entry["returncode"] = result.get("returncode")
    entry["routed_to"] = self._local_route_merge_rework(
        pr_number,
        issue_number,
        record,
        decision,
        reason="suite_failed",
        note=(
            "The full test suite failed on your branch after the base "
            f"merge ({' '.join(argv)}, exit {result.get('returncode')}). "
            "The code changes are already approved; fix the failing "
            "tests. Tail of the suite output:\n\n"
            f"```\n{tail}\n```"
        ),
    )
    return False


def _local_gate_launch(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    decision: dict[str, Any],
    reason: str,
    resync_count: int = 0,
    orphan_count: int = 0,
) -> bool:
    """Sync-merge the base, then spawn the detached suite runner.

    The base-sync merge is fast git plumbing and stays inline; everything
    after it (attach, spawn, claim write) returns in well under a second.
    Returns True only when a suite is actually in flight.
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    argv = suite_command_argv(self.config.dispatch.test_command, self.repo_root)
    if argv is None:
        # No resolvable suite command: fail closed. A lane that cannot
        # verify must not merge -- escalate mechanically so a human can
        # either configure a runner or merge by hand.
        entry["outcome"] = "error"
        entry["detail"] = "no test command resolvable for full-suite gate"
        self._local_merge_error_escalate(
            pr_number,
            issue_number,
            branch,
            "no test command resolvable for the local full-suite gate",
        )
        return False

    worktree_path = ensure_branch_worktree(self.repo_root, branch, self._layout.worktrees)
    if worktree_path is None:
        entry["outcome"] = "error"
        entry["detail"] = f"could not attach a worktree for {branch!r}"
        return False

    try:
        conflict = _merge_update_rework_branch(
            self.repo_root,
            worktree_path,
            branch,
            base_ref,
            self.config.dispatch.injected_paths,
            self.config.dispatch.materialize_dirs,
        )
    except Exception as exc:  # ReworkBranchConflictError / RuntimeError
        entry["outcome"] = "error"
        entry["detail"] = f"base sync merge failed: {exc}"
        self._local_merge_error_escalate(
            pr_number, issue_number, branch, f"base sync merge failed: {exc}"
        )
        return False
    if conflict is not None:
        entry["outcome"] = "conflict"
        entry["conflicted_paths"] = list(conflict.conflicted_files)
        entry["routed_to"] = self._local_route_merge_rework(
            pr_number,
            issue_number,
            record,
            decision,
            reason="merge_conflict",
            note=(
                f"Merging the base branch {base_ref!r} into {branch!r} "
                f"conflicted ({len(conflict.conflicted_files)} path(s)). "
                "Merge the base into your branch and resolve the conflicts. "
                "The code changes are already approved; do not re-litigate "
                "the review."
            ),
        )
        return False

    gate_head = branch_head_sha(self.repo_root, branch)
    gate_base = resolve_ref_sha(self.repo_root, base_ref)
    paths = local_suite_runner.suite_gate_paths(self.paths.dispatches, pr_number)
    launch = local_suite_runner.launch_suite_gate(
        worktree_path,
        argv,
        paths=paths,
        head_sha=gate_head or "",
        base_sha=gate_base,
    )
    if not launch.ok:
        detail = f"suite runner failed to spawn: {launch.error}"
        entry["outcome"] = "error"
        entry["detail"] = detail
        self._local_merge_error_escalate(pr_number, issue_number, branch, detail)
        return False

    self._local_gate_update(
        pr_key,
        {
            "local_suite_pid": launch.pid,
            "local_suite_process_start_time": launch.process_start_time,
            "local_suite_started_at": _wf.utc_now(),
            "local_suite_log": str(paths.log),
            "local_suite_gate_dir": str(paths.gate_dir),
            "local_suite_head": gate_head,
            "local_suite_base_sha": gate_base,
            "local_suite_argv": list(argv),
            "local_suite_resync_count": resync_count,
            "local_suite_orphan_count": orphan_count,
        },
        event=(
            "local_suite_launched",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "pid": launch.pid,
                "head_sha": gate_head,
                "base_sha": gate_base,
                "argv": list(argv),
                "log": str(paths.log),
                "reason": reason,
                "resync_count": resync_count,
                "orphan_count": orphan_count,
            },
        ),
    )
    entry["outcome"] = "suite_launched"
    entry["pid"] = launch.pid
    entry["reason"] = reason
    entry["suite_log"] = str(paths.log)
    return True


def _local_gate_finalize_merge(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    live_head: str,
    decision: dict[str, Any],
) -> None:
    """Green suite + stable pairing: advance the base and run terminal bookkeeping.

    Same merge outcomes the synchronous gate produced -- deferred / conflict /
    error / merged -- only reached now from a later pass, after the result
    file proved the tested pairing.
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    worktrees_dir = self._layout.worktrees
    outcome = merge_branch_into_base(self.repo_root, base_ref, branch, worktrees_dir=worktrees_dir)
    if outcome.status == "deferred":
        entry["outcome"] = "deferred"
        entry["detail"] = outcome.detail
        self._local_gate_event(
            "local_merge_deferred",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "detail": outcome.detail,
            },
        )
        return
    if outcome.status == "conflict":
        entry["outcome"] = "conflict"
        entry["conflicted_paths"] = list(outcome.conflicted_paths)
        entry["routed_to"] = self._local_route_merge_rework(
            pr_number,
            issue_number,
            record,
            decision,
            reason="merge_conflict",
            note=(
                f"Merging {branch!r} into the base branch {base_ref!r} "
                f"conflicted ({len(outcome.conflicted_paths)} path(s)). "
                "Merge the base into your branch and resolve the "
                "conflicts. The code changes are already approved; do not "
                "re-litigate the review."
            ),
        )
        return
    if outcome.status == "error":
        entry["outcome"] = "error"
        entry["detail"] = outcome.detail
        self._local_merge_error_escalate(pr_number, issue_number, branch, outcome.detail)
        return

    # Merged (or already contained): terminal bookkeeping.
    gate_head = record.get("local_suite_head")
    merged_sha = outcome.merged_sha or gate_head or live_head
    entry["outcome"] = "merged" if outcome.status == "merged" else "already_merged"
    entry["fast_forward"] = outcome.fast_forward
    entry["merged_sha"] = merged_sha
    # force=True like every other worker-worktree teardown: the branch
    # content is committed to base by construction, so remaining
    # uncommitted material is worker scratch (.venv, build output,
    # worker-tmp) -- a plain remove would refuse on a real .venv dir and
    # leak a worktree per merged issue. ``branch`` also deletes the
    # merged branch ref when auto_merge.delete_branch is configured,
    # matching the remote lane's post-merge cleanup.
    # Issue #1476: a worktree outside the managed dir is a foreign
    # checkout (``ensure_branch_worktree`` returns the registered path
    # wherever the branch happens to live) -- it belongs to whoever
    # created it and is never removed; the branch ref can't be deleted
    # while checked out anyway.
    worktree_path = worktree_for_branch(self.repo_root, branch)
    if worktree_path is not None and contains(worktrees_dir, worktree_path):
        remove_worktree(
            self.repo_root,
            worktree_path,
            force=True,
            branch=(branch if self.config.auto_merge.delete_branch else None),
        )
    remove_review_checkout(self.repo_root, pr_number, reviews_dir=self._layout.reviews_dir)
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(pr_key, {})
        state["prs"][pr_key] = {
            **without_review_dispatch_claim(pr_state),
            "number": pr_number,
            "local": True,
            "status": "merged",
            "merged_at": _wf.utc_now(),
            "merged_sha": merged_sha,
            "merge_method": "ff" if outcome.fast_forward else "no-ff",
            # Issue #1972: the merge-gate rework episode is resolved --
            # both per-kind attempt counters restart from zero. They
            # deliberately survive packet re-mints and rework routing;
            # this write plus the unescalate/de-escalation reset maps
            # are the only resets.
            "local_merge_conflict_rework_attempts": 0,
            "local_suite_failed_rework_attempts": 0,
            "local_merge_rework_reason": None,
        }
        issue_entry = state["issues"].get(str(issue_number), {})
        state["issues"][str(issue_number)] = _wf._merged_issue_fields(issue_entry, issue_number)
        state = self._record_event(
            state,
            "merge_succeeded",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "local": True,
                "fast_forward": outcome.fast_forward,
                "merged_sha": merged_sha,
                "detail": outcome.detail,
            },
        )
        self.write_gate.save_state(state)
    transition_result = self.write_gate.transition(
        self.gh, self.config.labels, issue_number, "merged"
    )
    if transition_result.outcome != TransitionOutcome.APPLIED:
        entry["label_error"] = {
            "edge": "merged",
            "outcome": transition_result.outcome.value,
        }
    # The issue itself closes on merge, same as a merged remote PR.
    try:
        self.gh.close_issue(issue_number)
    except Exception:
        entry["close_error"] = True


def _local_merge_error_escalate(
    self, pr_number: int, issue_number: int, branch: str, detail: str
) -> None:
    """Escalate a merge-gate infrastructure failure (never a silent merge)."""
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        # Local-bound "escalated" (issue #750 guard): the issue half runs
        # through ``_escalate_issue`` just below; this is the PR-record half.
        status = "escalated"
        state["prs"][str(pr_number)] = {
            **(state["prs"].get(str(pr_number)) or {}),
            "status": status,
            "escalation_reason": "local_merge_error",
        }
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason="local_merge_error",
            reason_class="mechanical",
            issue_extra={"merge_alert": detail},
        )
        state = self._record_event(
            state,
            "local_merge_failed",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "branch": branch,
                "detail": _wf._truncate_for_event(detail),
            },
            level="error",
        )
        self.write_gate.save_state(state)
    self.write_gate.transition(
        self.gh,
        self.config.labels,
        issue_number,
        _wf._escalation_edge("escalated", "mechanical"),
    )
