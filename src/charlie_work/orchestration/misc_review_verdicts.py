"""Review-verdict reaping delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 batch 3 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``.

Two Tier D names -- ``state_lock`` and ``remove_review_checkout`` -- are patched
on the ``charlie_work.workflow`` namespace by the suite, so the moved body
reaches them through ``_wf.`` to keep those seams landing. Every other free name
is imported directly from its defining module. The sibling calls
``self._comment_pr``, ``self._read_packet_head_oid`` and ``self.record_review``
stay ``self.`` calls (they resolve on the installed class).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from pathlib import Path
from typing import Any

from charlie_work.claude_code import _events_path
from charlie_work.harnesses import REVIEWER_ADAPTER_KINDS
from charlie_work.process_utils import find_worker_terminal_status
from charlie_work.verdict_parsing import (
    REVIEW_MISS_TURN_LIMIT,
    _extract_review_session_summary,
    _parse_review_verdict_from_events,
    _parse_review_verdict_from_files,
    _parse_review_verdict_from_log,
    _reviewer_session_metrics,
)
from charlie_work.worker import iter_workers


def _reap_review_verdicts(self, reviews_dir: Path) -> dict[str, Any]:
    """Record verdicts for dead reviewers whose sidecar log contains a valid
    fenced JSON verdict block.

    Iterates review sidecars for every review-capable harness (issue
    #1513: any ``adapter_kind`` in ``harnesses.REVIEWER_ADAPTER_KINDS`` --
    currently claude-code, devin ("devin-shell" harness), and api). For
    each reviewer that is no longer alive and still has a
    ``review_dispatch_dispatched`` claim, parse the log. If a valid
    verdict block is found, call ``record_review`` in-process with the
    packet head pinned as ``reviewed_head`` so the verdict is attributed
    to the diff the reviewer actually read. On success ``record_review``
    moves the PR to ``review_dispatch_completed``. If parsing fails or
    the verdict is malformed, the PR is left for
    ``_detect_and_handle_stalled_reviews`` to retry/backoff using the
    existing stale-claim path. Verdict extraction itself
    (``_parse_review_verdict_from_log``) is already generic (a plain
    fenced-JSON scan, falling back to a Claude-stream-json-specific
    decoder only on failure) -- this loop's own harness filter was the
    one place still hardcoded to claude-code.

    Returns a dict with ``recorded`` and ``missed`` verdict info lists for
    the dispatch result and the fleet attention digest.
    """
    recorded: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []

    for w in iter_workers(reviews_dir):
        if w.adapter_kind not in REVIEWER_ADAPTER_KINDS:
            continue
        if w.is_alive():
            continue

        pr_number = w.issue_number
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            if pr_state.get("review_dispatch_status") != "review_dispatch_dispatched":
                continue
            issue_number = pr_state.get("issue_number")

        verdict_source = "log"
        verdict = _parse_review_verdict_from_log(Path(w.log_path))
        if verdict is None:
            # Fallback: parse the structured events.jsonl. The plaintext log
            # may be truncated or the verdict block split across tee buffer
            # boundaries, but the stream-json events contain the assistant's
            # message text in discrete JSONL lines.
            events_path = _events_path(reviews_dir, pr_number, review=True)
            verdict_source = "events"
            verdict = _parse_review_verdict_from_events(events_path)
        if verdict is None:
            # Last resort (issue #566): the reviewer may have written its
            # verdict to a Markdown file it referenced in final output
            # instead of re-emitting the fenced block. mtime-gated to this
            # session's started_at so stale files never resurrect old
            # verdicts.
            file_hit = _parse_review_verdict_from_files(
                Path(w.log_path),
                self.paths.prs / f"pr-{pr_number}",
                w.started_at,
            )
            if file_hit is not None:
                verdict, file_source = file_hit
                verdict_source = f"file:{file_source}"
        if verdict is None:
            # No structured verdict found. Before discarding this reviewer's
            # work, check if it did substantial analysis (e.g. hit the
            # --max-turns limit) and post a summary PR comment so the work
            # is not silently lost. Only post once per dispatch lifecycle.
            pr_state_dict = pr_state
            # One-shot guard. ``review_miss_summary_posted`` covers every
            # miss reason; the legacy turn-limit key is still honoured so
            # PRs mid-lifecycle when this shipped don't get a second
            # comment.
            already_posted = pr_state_dict.get("review_miss_summary_posted") or pr_state_dict.get(
                "review_turn_limit_summary_posted"
            )
            if not already_posted:
                events_path = _events_path(reviews_dir, pr_number, review=True)
                max_turns = self.config.review_dispatch.review_max_turns
                # Issue #1354: read the reviewer's exit code from the
                # durable terminal-status record (written by
                # ``start_terminal_status_watcher`` in
                # ``launch_claude_worker``) so the terminating-cause
                # extractor can fold it into the ``cause`` field of the
                # ``review_verdict_missed`` payload. The terminal-status
                # watcher is now started for review launches too (see
                # ``claude_code.launch_claude_worker``); for sessions
                # that died before that change landed, the record is
                # absent and ``exit_code`` stays ``None``, in which case
                # ``_extract_terminating_cause`` falls back to the
                # stream-json result event or the explicit
                # ``{"cause": "unknown"}`` sentinel.
                #
                # The reviewer's terminal-status file is written under
                # ``reviews_dir`` keyed by ``pr_number`` -- the review
                # launch site (``dispatch_reviews``) calls
                # ``launch_claude_worker(sessions_dir=reviews_dir,
                # issue_number=pr_number, review=True)``, and
                # ``start_terminal_status_watcher`` writes to
                # ``worker_terminal_status_path(sessions_dir,
                # issue_number, ...)``. Reading from the layout's default
                # sessions dir keyed by the linked issue number would
                # return the original coding worker's stale exit code,
                # not the reviewer's.
                terminal = find_worker_terminal_status(reviews_dir, pr_number)
                exit_code = terminal.get("exit_code") if terminal else None
                outcome = _extract_review_session_summary(
                    events_path,
                    Path(w.log_path),
                    max_turns,
                    session_limit_markers=self.config.runtime.session_limit_markers,
                    exit_code=exit_code,
                )
                if outcome is not None:
                    try:
                        self._comment_pr(pr_number, outcome.text)
                    except Exception:
                        pass
                    with _wf.state_lock(self.paths.state_file):
                        state = _wf.load_state(self.paths.state_file)
                        ps = state["prs"].get(str(pr_number), {})
                        # The turn-limit key means "this dispatch
                        # lifecycle's session did substantial work then
                        # died". The provider-throttle sweep reads it to
                        # deny a rollback (#583) and #584 counts it as a
                        # turn-limit death, so a session that never
                        # reached turn 1 must not set it -- that death is
                        # environmental and says nothing about this PR.
                        # Issue #1439: a turn-limit death increments the
                        # per-PR miss streak so the NEXT dispatch's cap
                        # escalates one step instead of relaunching with
                        # the identical flat budget. Non-turn-limit deaths
                        # (launch_failed, died_mid_session) do NOT
                        # increment -- those have disjoint remediations and
                        # counting them here would conflate a quota/launch
                        # outage with a genuine "reviewer ran out of
                        # navigation budget" signal.
                        _is_turn_limit_miss = outcome.reason == REVIEW_MISS_TURN_LIMIT
                        _prior_streak = int(ps.get("review_turn_limit_miss_streak", 0))
                        _new_streak = _prior_streak + 1 if _is_turn_limit_miss else _prior_streak
                        state["prs"][str(pr_number)] = {
                            **ps,
                            "review_miss_summary_posted": True,
                            "review_turn_limit_summary_posted": (outcome.did_substantial_work),
                            "review_turn_limit_miss_streak": _new_streak,
                        }
                        state = _wf.append_event(
                            state,
                            "review_verdict_missed",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "reason": outcome.reason,
                                "turn_count": outcome.turn_count,
                                "tool_call_count": outcome.tool_call_count,
                                # Issue #1354: the terminating-cause
                                # dict (exit code, stream-json result
                                # event fields, or ``{"cause": "unknown"}``
                                # when nothing could be captured). Always
                                # present so a downstream query can
                                # distinguish "no cause captured" from
                                # "cause field absent".
                                "cause": outcome.terminating_cause,
                                # Issue #1439: mirror the post-increment
                                # streak into the event so a downstream
                                # query can reconstruct the cap escalation
                                # history without a join back to state.
                                "turn_limit_miss_streak": _new_streak,
                            },
                            state_path=self.paths.state_file,
                        )
                        _wf.save_state(self.paths.state_file, state)
                    missed.append(
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "reason": outcome.reason,
                            "cause": outcome.terminating_cause,
                        }
                    )
                    # Issue #1354: release this dead reviewer's isolated
                    # review checkout in the SAME pass that detected the
                    # death, rather than waiting for
                    # ``_detect_and_handle_stalled_reviews``'s 5-minute
                    # stale-claim timeout. The reviewer is already dead
                    # (``w.is_alive()`` returned False at the top of this
                    # loop) and no verdict was found, so the checkout is
                    # pure overhead -- a detached-HEAD git worktree that
                    # holds no recoverable work. Releasing it here means
                    # the claim directory does not linger for a full
                    # stale-claim interval.
                    #
                    # The sidecar is deliberately NOT reaped here: the
                    # stalled-review sweep (which runs immediately after
                    # this function in ``dispatch_reviews``) reads the
                    # sidecar's log tail to classify provider-throttle
                    # signatures and arm fleet-wide backoff. Reaping the
                    # sidecar here would prevent that classification,
                    # silently disarming the quota-exhaustion backoff for
                    # a death that may have been caused by exactly that
                    # condition. The stalled sweep reaps the sidecar
                    # itself after classification (or on its next pass
                    # once the stale timeout elapses).
                    _wf.remove_review_checkout(self.repo_root, pr_number, reviews_dir=reviews_dir)
            continue

        packet_head_sha = self._read_packet_head_oid(pr_number)
        session_metrics = _reviewer_session_metrics(
            _events_path(reviews_dir, pr_number, review=True), verdict_source
        )
        # Fold the review_effort experiment's arm/effort assignment (set
        # at claim time in dispatch_reviews) into session_metrics so the
        # record_review event alone is enough to split spend/quality by
        # arm, without a join back to the dispatch event.
        review_effort_arm = pr_state.get("review_effort_arm")
        review_effort_used = pr_state.get("review_effort_used")
        if review_effort_arm is not None or review_effort_used is not None:
            session_metrics = {
                **(session_metrics or {}),
                "review_effort_arm": review_effort_arm,
                "review_effort_used": review_effort_used,
            }
        result = self.record_review(
            pr_number,
            verdict["decision"],
            summary=verdict["summary"],
            reviewed_head=packet_head_sha,
            required_changes=verdict["required_changes"],
            session_metrics=session_metrics,
            verdict_provenance="fresh_llm_review",
            # Issue #1340: thread the parser-level provenance into the
            # decision file so a later reader can distinguish a
            # log-extracted verdict (dead reviewer) from a clean
            # structured completion. ``verdict_source`` is the same value
            # already threaded into ``session_metrics`` above.
            verdict_source=verdict_source,
        )
        if result.ok:
            recorded.append(
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "decision": verdict["decision"],
                    "verdict_source": verdict_source,
                }
            )
        else:
            reason = result.message or "record_review failed"
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                state = _wf.append_event(
                    state,
                    "review_verdict_missed",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "reason": reason,
                    },
                    state_path=self.paths.state_file,
                )
                _wf.save_state(self.paths.state_file, state)
            missed.append(
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "reason": reason,
                }
            )

    return {"recorded": recorded, "missed": missed}
