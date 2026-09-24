# Worker Rework Task: issue #$issue_number (local repository, no remote)

Your branch requires changes before it can be approved. **This repository has
no remote.** There is no pull request and nothing to push to — your
deliverable is new commits on your existing branch. Where a shared section
below says "the PR body", read "your final commit message"; where it says
"before pushing", read "before your final commit".

## Branch under rework

- Issue: #$issue_number
- Title: $pr_title
- Branch (already checked out in your worktree — commit only here): `$branch_name`

## Orchestrator review

$dispatch_note_block

$required_changes_section

## No operator decisions in this brief

This brief carries NO operator decisions about the review findings — not in
the orchestrator note, and not in the findings above. If a required change
offers options contingent on an operator/human call ("either accept as-is, or
defer them", "confirm explicitly", "needs sign-off"), that call has NOT been
made and you must not make it for them.

- Never claim, infer, or record that the operator or any human approved,
  declined, accepted, deferred, or chose anything — not in a commit message,
  a comment, or code.
- If a required change needs a human/operator decision, STOP. Do not pick
  an option yourself and do not rework around it. Write
  `.worker-outcome.json` in the repository root with this exact shape:

  ```json
  {"outcome": "blocked", "reason_kind": "ambiguous_scope", "detail": "<the required-change item that needs a human decision>"}
  ```

  Then exit cleanly without committing further work. The orchestrator records
  the blocked outcome in your session's terminal status and the unchanged head
  is escalated to a human — the operator decides, you do not decide for them.

## Required behavior

- Act on the review above — the orchestrator note and any findings — before
  anything else; that is the reason this branch was sent back for rework.
  Where the findings carry severities, Critical and Important items are
  mandatory; a Minor item may be skipped only if you say so explicitly in the
  final commit message.
- Stay on the existing branch `$branch_name`. Do not create a new branch
  unless the branch is unrecoverable.
- If your branch is behind its base or the merge check reported a conflict,
  merge the base branch into your branch and resolve any conflicts.
- Preserve the original issue scope.
- Add or update tests for the review findings.
- Before your final commit, match the repository's linters locally —
  `ruff check`, `ruff format --check`, and, if the repository has a
  `.pre-commit-config.yaml`, `pre-commit run` on the files you touched —
  and COMMIT anything they fix.
- Re-run verification and record the results in your final commit message.
- If you disagree with a finding, explain with evidence in the commit message
  instead of ignoring it.

$section_execution_contract

$section_repo_commands_precedence

$section_scope_contract

$section_invariant_enumeration

$section_caller_sweep

$section_local_no_merge_contract

$section_process_lifetime

$section_session_scratch_dir

$section_mutation_check

$section_test_hygiene

## Done condition

You are done only when the branch `$branch_name` has new commits addressing
the review and the working tree is clean:

```bash
git status --short          # must print nothing you authored
git log --oneline -5        # your rework commit(s) on top of the prior head
```

Do not push. Do not open a pull request. Do not merge. After your final commit,
exit cleanly — the orchestrator detects the advanced head, rebuilds the review
packet, and re-reviews the branch. Uncommitted work is invisible to it and is a
task FAILURE.
