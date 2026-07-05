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
- `/test` - Run the test suite and verify all tests pass (only if it wraps the canonical command below)
- `/preflight` - Match CI (ruff + ruff-format + pre-commit) before pushing
- `/push` - Push the branch to GitHub
- `/create-pr` - Create a pull request with proper formatting
- `/complete` - Finalize the session and verify cleanup

## Required implementation loop

1. Use `/create-branch` to ensure you're on the correct branch.
2. Read `CLAUDE.md`, `CONTRIBUTING.md`, the issue, and relevant code.
3. Reproduce or precisely explain the defect/requirement.
4. Implement the smallest correct change.
5. Run the full test suite with the canonical command from the worktree root:
   ```bash
   uv run --extra dev pytest -q --tb=short
   ```
   The `/test` skill is a convenience shortcut but may not run the full suite — always
   use the explicit command for final verification. Quote the exact command you ran
   AND the collected/passed count in your completion report (e.g., "300 collected, 300 passed").
6. Add or update regression tests unless not applicable.
7. Use `/commit` to commit your changes with conventional format.
8. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it
   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
9. Use `/push` to push your branch to GitHub.
10. Use `/create-pr` to create a pull request with proper formatting.
11. Use `/complete` to finalize the session.

## PR requirements

- Title format: Conventional-Commits format (`type(scope): description`).
  - Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
  - The issue link goes in the body only (see below), not the title.
  - If the issue title has a conventional-commit prefix, mirror it in the PR title.
  <!-- JANITOR_TITLE_EXAMPLE: fix(janitor): align worker template with conventional-commit requirements -->
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
3. After verifying the push, re-read your PR body and make every claim literally true at the pushed head: the suite count must come from your final local run on the pushed tree, file/occurrence lists must match the final diff exactly, and any carve-outs or partial applications must be disclosed as such. Update the body with `gh pr edit` if anything is stale. A PR body with a false or stale claim fails review.

Only when the PR head points at your pushed commit is the task complete.
