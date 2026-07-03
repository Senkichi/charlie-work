# Devin Worker Rework Task: PR #$pr_number

Your PR requires changes before it can be approved.

## PR

- Number: #$pr_number
- Title: $pr_title
- URL: $pr_url
- Linked issue: #$issue_number

## Orchestrator review

```md
$review_summary
```

## Required behavior

- First, merge the PR's base branch (e.g., `origin/main` unless the PR targets another base)
  into your branch to incorporate any base changes that landed since the branch was created.
  This catches skew even when auto-update is off.
- Update the existing PR. Do not open a new PR unless the branch is unrecoverable.
- Address every required change directly.
- Preserve the original issue scope.
- Add or update tests for the review findings.
- Before pushing, run `/preflight` (ruff + ruff-format + pre-commit) and COMMIT anything
  it fixes. Pushing to an existing PR is gated the same as opening one — a CI-dirty tree
  (uncommitted reflow, un-normalized fixture) will be blocked.
- Re-run verification and update the PR body or comment with results.
- If you disagree with a finding, explain with evidence in the PR instead of ignoring it.

## Done condition

You are done only when the existing PR has new commits addressing the review and the PR includes updated verification evidence.

## FINAL STEP — push and verify

**Committing locally is NOT done.** The PR head must advance to reflect your work.

After your final commit:

1. Run the full test suite before pushing:
   ```bash
   uv run --extra dev pytest -q --tb=short
   ```
2. Push your branch:
   ```bash
   git push origin $branch_name
   ```
3. Verify the PR head advanced:
   ```bash
   gh pr view $pr_number --json headRefOid
   ```
   Confirm the returned `headRefOid` equals `git rev-parse HEAD`.

Only when the PR head points at your pushed commit is the rework complete.
