# Devin Worker Task: Issue #$issue_number

You are a worker agent assigned to exactly one GitHub issue.

## Issue

- Number: #$issue_number
- Title: $issue_title
- URL: $issue_url
- Model tier target: $worker_model_tier

## Branch

Create and use this branch:

```text
$branch_name
```

## Issue body

```md
$issue_body
```

## Scope contract

- Solve only issue #$issue_number.
- Do not batch unrelated fixes.
- Do not perform opportunistic refactors.
- Preserve the patterns and invariants in `CLAUDE.md`.
- If the issue is ambiguous, stop and explain the blocker instead of guessing.
- If the fix touches security-sensitive behavior, call it out explicitly in the PR.

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
