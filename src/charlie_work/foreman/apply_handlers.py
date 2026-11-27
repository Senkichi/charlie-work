"""The per-action-type handlers of the Apply engine (issue #3119).

Split out of ``apply.py``: one ``_<type>`` function per action type, each
returning a ``_Verdict``, plus ``_process`` — the dispatcher ``apply_turn``
calls once per action. Every handler runs the same pipeline: charter repo
authority → bounds → type-specific checks → effect — and a refusal is a
verdict value, never an exception.
"""

from __future__ import annotations

from ..github import label_names
from ..github_body_scan import parse_blockers
from ..issue_contract_lint import lint_issue_body
from ..labels import apply_issue_labels
from .actions import (
    Action,
    AmendIssueAction,
    CommentIssueAction,
    DeleteMemoryAction,
    FileIssueAction,
    ProposeDoneAction,
    RaiseItemAction,
    WithdrawIssueAction,
    WriteMemoryAction,
)
from .apply_checks import (
    _arm_or_raise,
    _blocker_tail,
    _check_outbound_and_slug,
    _has_comment,
    _live_authored,
    _post_comment,
    _record_authored,
    _repo_check,
)
from .apply_context import (
    _OWNER_LABEL_COLOR,
    OUTCOME_REFUSED,
    _applied,
    _body_hash,
    _marker,
    _marker_pattern,
    _refused,
    _Run,
    _Verdict,
)
from .items import create_item
from .memory import delete_memory as _delete_memory_file
from .memory import write_memory as _write_memory_file
from .paths import state_path
from .state import read_state_snapshot


def _file_issue(run: _Run, action: FileIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)

    # Idempotent create runs before the bounds: a crash between create and
    # record replays into this search, and the already-filed issue must
    # report applied — a bound refusal here would falsely cascade "the
    # action for key k was refused" to every dependent.
    existing_number = run.marker_map.get((run.turn_id, action.key))
    if existing_number is not None:
        if run.dry_run:
            return _applied(
                "would reuse existing issue "
                f"#{existing_number} ({_arm_or_raise(run, existing_number, action.key, action.title)})",
                issue_number=existing_number,
            )
        live = run.gh.issue_view(existing_number)
        _record_authored(run, existing_number, str(live.get("body") or ""))
        detail = f"reused existing issue #{existing_number}"
        if str(live.get("state", "")).upper() == "OPEN":
            detail += f", {_arm_or_raise(run, existing_number, action.key, action.title)}"
        run.key_numbers[action.key] = existing_number
        run.file_issues_applied += 1
        return _applied(detail, issue_number=existing_number)

    if action.key in run.duplicate_file_keys:
        return _refused(
            f"key {action.key!r}: duplicate file_issue key — markers and "
            "blocked_by refs must resolve to exactly one action"
        )
    bounds = run.charter.bounds
    if run.open_issues_used >= bounds.max_open_issues:
        return _refused(
            f"bound max_open_issues: {run.open_issues_used} open issues already "
            f"carry {run.owner_label} (limit {bounds.max_open_issues})"
        )
    if run.file_issues_applied >= bounds.max_issues_per_turn:
        return _refused(
            f"bound max_issues_per_turn: {run.file_issues_applied} file_issue "
            f"actions already applied this Turn (limit {bounds.max_issues_per_turn})"
        )

    blocker_numbers: list[int] = []
    for ref in action.blocked_by:
        if ref.startswith("key:"):
            dep = ref[4:]
            if dep in run.key_numbers:
                blocker_numbers.append(run.key_numbers[dep])
            elif dep in run.failed_keys:
                return _refused(f"blocked_by {ref!r}: the action for key {dep!r} was refused")
            elif dep in run.cycle_keys:
                return _refused(
                    f"blocked_by {ref!r}: key {dep!r} is in a dependency cycle, refused"
                )
            elif dep in run.known_file_keys:
                return _refused(f"blocked_by {ref!r}: key {dep!r} was not applied")
            else:
                return _refused(
                    f"blocked_by {ref!r}: no file_issue action in this Turn has key {dep!r}"
                )
        else:
            blocker_numbers.append(int(ref[1:]))

    problems = lint_issue_body(action.body, frozenset())
    if problems:
        return _refused("body fails the issue contract: " + "; ".join(problems))

    refusal = _check_outbound_and_slug(
        run,
        "create_issue",
        (("title", action.title), ("body", action.body)),
        issue_number=None,
    )
    if refusal is not None:
        return _refused(refusal)

    blocker_set = frozenset(blocker_numbers)
    marker = _marker(run.charter.name, run.turn_id, action.key)
    final_body = "\n\n".join(
        part for part in (action.body.strip(), _blocker_tail(blocker_set, marker)) if part
    )
    problems = lint_issue_body(final_body, blocker_set)
    if problems:
        return _refused("composed body fails the issue contract: " + "; ".join(problems))

    if run.dry_run:
        arming = _arm_or_raise(run, 0, action.key, action.title)
        return _applied(f"would file {action.title!r} ({arming})")

    known_labels = {entry.get("name") for entry in run.gh.label_list()}
    if run.owner_label not in known_labels:
        run.gh.label_create(
            run.owner_label,
            _OWNER_LABEL_COLOR,
            f"Authored by foreman {run.charter.name}",
        )
    number = run.gh.create_issue(action.title, final_body, labels=[run.owner_label])
    if number is None:
        return _refused("issue create failed (guard refusal or write failure)")
    run.marker_map[(run.turn_id, action.key)] = number
    live = run.gh.issue_view(number)
    _record_authored(run, number, str(live.get("body") or final_body))
    arming = _arm_or_raise(run, number, action.key, action.title)
    run.key_numbers[action.key] = number
    run.file_issues_applied += 1
    run.open_issues_used += 1
    return _applied(f"filed #{number}, {arming}", issue_number=number)


def _amend_issue(run: _Run, action: AmendIssueAction) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    issue, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)

    recorded = read_state_snapshot(run.foreman_root).issues.get(action.number)
    live_body = str(issue.get("body") or "")
    if recorded is None:
        return _refused(
            f"issue #{action.number}: no authored record in state.json — "
            "the body hash cannot be checked"
        )
    if _body_hash(live_body) != recorded.body_hash:
        return _refused(
            f"issue #{action.number}: live body hash differs from the recorded "
            "hash — the operator edited it and their text is never overwritten"
        )

    problems = lint_issue_body(action.body, frozenset())
    if problems:
        return _refused("body fails the issue contract: " + "; ".join(problems))
    refusal = _check_outbound_and_slug(
        run, "edit_issue_body", (("body", action.body),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)

    blockers = frozenset(parse_blockers(live_body))
    marker_match = _marker_pattern(run.charter.name).search(live_body)
    marker = (
        marker_match.group(0) if marker_match else _marker(run.charter.name, run.turn_id, "amend")
    )
    new_body = "\n\n".join(
        part for part in (action.body.strip(), _blocker_tail(blockers, marker)) if part
    )
    problems = lint_issue_body(new_body, blockers)
    if problems:
        return _refused("composed body fails the issue contract: " + "; ".join(problems))

    if run.dry_run:
        return _applied(f"would amend #{action.number}", issue_number=action.number)

    # Labels, marker and openness are re-read live immediately before the
    # effect: an operator edit between check and write still refuses.
    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)
    if not run.gh.edit_issue_body(action.number, new_body):
        return _refused(f"issue #{action.number}: body edit failed")
    live = run.gh.issue_view(action.number)
    _record_authored(run, action.number, str(live.get("body") or new_body))
    return _applied(f"amended #{action.number}", issue_number=action.number)


def _withdraw_issue(run: _Run, action: WithdrawIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)

    comment = (
        f"Withdrawn by foreman `{run.charter.name}` in turn `{run.turn_id}`.\n\n{action.reason}"
    )
    refusal = _check_outbound_and_slug(
        run, "issue_comment", (("body", comment),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)

    if run.dry_run:
        return _applied(f"would withdraw #{action.number}", issue_number=action.number)

    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)
    # Per the design: the Ready label goes first, then the close carries the
    # reason as a comment — an unarmed-then-closed issue can never dispatch.
    # write-gate-exempt(issue=3119): the foreman Apply engine has no write_gate receiver; apply_issue_labels is itself the issue-#2226 label-write seam.
    label_result = apply_issue_labels(
        run.gh,
        run.labels,
        action.number,
        remove=(run.labels.ready,),
        state_path=state_path(run.foreman_root),
        repo=run.charter.repos[0] if run.charter.repos else None,
        cause="foreman_withdraw",
    )
    if not label_result.ok:
        return _refused(f"issue #{action.number}: could not remove {run.labels.ready}")
    comment_failure = _post_comment(run, action.number, index, comment)
    if comment_failure is not None:
        return _refused(comment_failure)
    if not run.gh.close_issue(action.number):
        return _refused(f"issue #{action.number}: close failed")
    return _applied(f"withdrew #{action.number}", issue_number=action.number)


def _comment_issue(run: _Run, action: CommentIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    issue = run.gh.issue_view(action.number)
    if not issue:
        return _refused(f"issue #{action.number}: not found in {run.charter.repos[0]}")
    if run.owner_label not in label_names(issue):
        return _refused(f"issue #{action.number}: lacks ownership label {run.owner_label!r}")
    refusal = _check_outbound_and_slug(
        run, "issue_comment", (("body", action.body),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)
    if run.dry_run:
        return _applied(f"would comment on #{action.number}", issue_number=action.number)
    if _has_comment(run, action.number, action.body):
        return _applied(
            f"comment already present on #{action.number} (deduped)",
            issue_number=action.number,
        )
    failure = _post_comment(run, action.number, index, action.body)
    if failure is not None:
        return _refused(failure)
    return _applied(f"commented on #{action.number}", issue_number=action.number)


def _raise_item(run: _Run, action: RaiseItemAction) -> _Verdict:
    if action.key in run.duplicate_item_keys:
        return _refused(
            f"key {action.key!r}: duplicate raise_item key — item ids are "
            "derived from the key and must be unique per Turn"
        )
    item_id = f"{run.turn_id}:{action.key}"
    if run.dry_run:
        return _applied(f"would raise {action.kind} item {item_id}", item_id=item_id)
    # For a report the question field carries the report body; options and
    # recommended are forwarded regardless so the item store's own per-kind
    # validation refuses a shape that doesn't fit the kind.
    result = create_item(
        run.foreman_root,
        kind=action.kind,
        question=action.question if action.kind != "report" else None,
        text=action.question if action.kind == "report" else None,
        options=action.options,
        recommended=action.recommended,
        command=action.command,
        item_id=item_id,
    )
    if not result.ok:
        if result.error and "already exists" in result.error:
            return _applied(f"item {item_id} already exists (deduped)", item_id=item_id)
        return _refused(f"item store refused: {result.error}")
    return _applied(f"raised {action.kind} item {result.value.id}", item_id=result.value.id)


def _write_memory(run: _Run, action: WriteMemoryAction) -> _Verdict:
    if run.dry_run:
        return _applied(f"would write memory {action.path!r}")
    result = _write_memory_file(run.foreman_root, action.path, action.content)
    if not result.ok:
        return _refused(f"memory refused: {result.error}")
    return _applied(f"wrote memory {action.path!r}")


def _delete_memory(run: _Run, action: DeleteMemoryAction) -> _Verdict:
    if run.dry_run:
        return _applied(f"would delete memory {action.path!r}")
    result = _delete_memory_file(run.foreman_root, action.path)
    if not result.ok:
        return _refused(f"memory refused: {result.error}")
    return _applied(f"deleted memory {action.path!r}")


def _propose_done(run: _Run, action: ProposeDoneAction, index: int) -> _Verdict:
    if run.charter.standing:
        return _refused("charter is standing: a standing foreman never proposes done")
    item_id = f"{run.turn_id}:propose-done:{index}"
    if run.dry_run:
        return _applied(f"would raise done-confirmation item {item_id}", item_id=item_id)
    result = create_item(
        run.foreman_root,
        kind="decision",
        question=f"Foreman {run.charter.name} proposes done. Is the outcome met?",
        text=action.evidence,
        options=("Done — mark the foreman done", "Not done — keep working"),
        recommended=0,
        item_id=item_id,
    )
    if not result.ok:
        if result.error and "already exists" in result.error:
            return _applied(f"item {item_id} already exists (deduped)", item_id=item_id)
        return _refused(f"item store refused: {result.error}")
    return _applied(f"raised done-confirmation item {result.value.id}", item_id=result.value.id)


def _process(run: _Run, action: Action, index: int) -> _Verdict:
    if isinstance(action, FileIssueAction):
        verdict = _file_issue(run, action, index)
        if verdict.outcome == OUTCOME_REFUSED:
            run.failed_keys.add(action.key)
        return verdict
    if isinstance(action, AmendIssueAction):
        return _amend_issue(run, action)
    if isinstance(action, WithdrawIssueAction):
        return _withdraw_issue(run, action, index)
    if isinstance(action, CommentIssueAction):
        return _comment_issue(run, action, index)
    if isinstance(action, RaiseItemAction):
        return _raise_item(run, action)
    if isinstance(action, WriteMemoryAction):
        return _write_memory(run, action)
    if isinstance(action, DeleteMemoryAction):
        return _delete_memory(run, action)
    if isinstance(action, ProposeDoneAction):
        return _propose_done(run, action, index)
    return _refused(f"unknown action type {type(action).__name__}")  # unreachable
