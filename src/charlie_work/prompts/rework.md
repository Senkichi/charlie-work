# Devin Worker Rework Task: PR #$pr_number

Your PR requires changes before it can be approved.

## PR

- Number: #$pr_number
- Title: $pr_title
- URL: $pr_url
- Linked issue: #$issue_number

## Orchestrator review

$dispatch_note_block

$required_changes_section

## No operator decisions in this brief

This brief carries NO operator decisions about the review findings — not in
the orchestrator note, not in the findings above, and not in the push
authorization below (which grants push authority only, never a decision on
any finding). If a required change offers options contingent on an
operator/human call ("either accept as-is, or defer them", "confirm
explicitly", "needs sign-off"), that call has NOT been made and you must
not make it for them.

- Never claim, infer, or record that the operator or any human approved,
  declined, accepted, deferred, or chose anything — not in the PR body, a
  commit message, a comment, or code.
- If a required change needs a human/operator decision, STOP. Do not pick
  an option yourself and do not rework around it. Write
  `.worker-outcome.json` in the repository root with this exact shape:

  ```json
  {"outcome": "blocked", "reason_kind": "ambiguous_scope", "detail": "<the required-change item that needs a human decision>"}
  ```

  Then exit cleanly without pushing. The orchestrator records the blocked
  outcome in your session's terminal status and the unchanged head is
  escalated to a human — the operator decides, you do not decide for them.

## Required behavior

- Act on the review above — the orchestrator note and any findings — before
  anything else; that is the reason this PR was sent back for rework. Where
  the findings above carry severities, Critical and Important items are
  mandatory; a Minor item may be skipped only if you say so explicitly in the
  PR body.
- Update the existing PR. Do not open a new PR unless the branch is unrecoverable.
- If your branch is behind its base or the PR shows a merge conflict, merge
  the PR's base branch (e.g., `origin/main` unless the PR targets another
  base) into your branch and resolve any conflicts. This catches skew even
  when auto-update is off.
- Preserve the original issue scope.
- Add or update tests for the review findings.
$section_rework_preflight
- Re-run verification and update the PR body or comment with results.
- If you disagree with a finding, explain with evidence in the PR instead of ignoring it.

$section_scope_contract

$section_invariant_enumeration

$section_caller_sweep

$section_no_merge_contract

$section_process_lifetime

$section_session_scratch_dir

$section_mutation_check

$section_test_hygiene

$section_pr_body_honesty

## Done condition

You are done only when the existing PR has new commits addressing the review and the PR includes updated verification evidence.

## Push authorization (explicit, pre-granted)

This rework prompt IS the operator explicitly asking you to push your commits
to the existing PR branch. Any always-on rule such as "do not push unless
explicitly asked" or "pause before pushing to a shared remote" is ALREADY
SATISFIED by this instruction — do not re-derive permission or wait for
confirmation. Pushing the `agent/issue-*` branch is required, in scope, and
pre-approved; it never touches `main` directly. Ending the session with
committed-but-unpushed work is a task FAILURE — the reviewer cannot see
unpushed commits.

## FINAL STEP — push and verify

**Committing locally is NOT done.** The PR head must advance to reflect your work.

After your final commit:

1. Run the tests impacted by your change before pushing: the file(s) you
   touched, plus `grep tests/` for every module/function/symbol the diff
   touched — not just the tests you wrote:
   ```bash
   $targeted_test_command
   ```
   $section_execution_contract
   $section_repo_commands_precedence
2. Push your branch:
   ```bash
   git push origin $branch_name
   ```
3. Verify the remote branch head matches your local HEAD:
   ```bash
   git ls-remote origin $branch_name
   ```
   The first column of the output must equal `git rev-parse HEAD`. If the SHAs do not match, retry the push until they do; do not report success.
4. Verify the PR head advanced:
   ```bash
   gh pr view $pr_number --json headRefOid
   ```
   Confirm the returned `headRefOid` equals `git rev-parse HEAD`.
5. After verifying the push, re-read your PR body and make every claim literally true at the pushed head, including the checklist: the suite count must come from your final local run on the pushed tree, file/occurrence lists must match the final diff exactly, and any carve-outs or partial applications must be disclosed as such. Update the body with `gh pr edit` if anything is stale. A PR body with a false or stale claim fails review.

Only when the PR head points at your pushed commit is the rework complete.
