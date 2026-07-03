# Devin Worker Task: Issue #$issue_number

You are a worker agent assigned to exactly one GitHub issue.

## Issue

$section_issue_metadata

## Branch

Create and use this branch:

```text
$branch_name
```

## Issue body

```md
$issue_body
```

$section_scope_contract

## Available skills

The following skills are available to help you complete this task:

- `/create-branch` - Ensure the branch is created and checked out
- `/commit` - Create a conventional commit with proper formatting
- `/test` - Run the test suite and verify all tests pass
- `/preflight` - Match CI (ruff + ruff-format + pre-commit) before pushing
- `/push` - Push the branch to GitHub
- `/create-pr` - Create a pull request with proper formatting
- `/complete` - Finalize the session and verify cleanup

## Required implementation loop

1. Use `/create-branch` to ensure you're on the correct branch.
2. Read `CLAUDE.md`, `CONTRIBUTING.md`, the issue, and relevant code.
3. Reproduce or precisely explain the defect/requirement.
4. Implement the smallest correct change.
5. Use `/test` to run the test suite and verify all tests pass.
6. Add or update regression tests unless not applicable.
7. Use `/commit` to commit your changes with conventional format.
8. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it
   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
9. Use `/push` to push your branch to GitHub.
10. Use `/create-pr` to create a pull request with proper formatting.
11. Use `/complete` to finalize the session.

## PR requirements

- Title format: `Fix #$issue_number: <short title>`.
- Body must include `Closes #$issue_number`.
- Fill out `.github/pull_request_template.md`.
- Include exact commands run and results.
- Include risks and any uncertain areas.

## Done condition

You are done only when the PR is open, linked to issue #$issue_number, and includes a clear verification summary.

**Committing locally is NOT done.** The PR must exist and point at your pushed head.

After your final commit:

1. Push your branch:
   ```bash
   git push origin $branch_name
   ```
2. Verify the PR exists and points at your commit:
   ```bash
   gh pr view $branch_name --json headRefOid
   ```
   Confirm the returned `headRefOid` equals `git rev-parse HEAD`.

Only when the PR head points at your pushed commit is the task complete.
