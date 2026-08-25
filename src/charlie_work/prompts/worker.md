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

$issue_body_block$issue_comments

$module_map
$section_scope_contract

$section_spec_letter

$section_no_merge_contract

$section_process_lifetime

$section_api_shape_validation

$section_caller_sweep

$section_parallel_investigation

$section_config_parity

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

   $section_invariant_enumeration
5. Run the tests impacted by your change from the worktree root: the test
   file(s) you added or modified, plus `grep tests/` for every
   module/function/symbol your production diff touched and run every
   matching test file — not just the tests you wrote:
   ```bash
   uv run --extra dev pytest tests/test_<touched_module>.py -q --tb=short
   ```
   $section_execution_contract
   The `/test` skill is a convenience shortcut but may not cover
   every impacted file — always use the explicit command for final
   verification.
6. Add or update regression tests unless not applicable.
7. $section_ruff_preflight
8. Use `/commit` to commit your changes with conventional format.
9. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it
   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
10. Use `/push` to push your branch to GitHub.
11. Use `/create-pr` to create a pull request with proper formatting.
12. Use `/complete` to finalize the session.

$section_mutation_check

$section_test_hygiene

$section_pr_body_honesty

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
2. Verify the remote branch head matches your local HEAD:
   ```bash
   git ls-remote origin $branch_name
   ```
   The first column of the output must equal `git rev-parse HEAD`. If the SHAs do not match, retry the push until they do; do not report success.
3. Verify the PR exists and points at your commit:
   ```bash
   gh pr view $branch_name --json headRefOid
   ```
   Confirm the returned `headRefOid` equals `git rev-parse HEAD`.
4. After verifying the push, re-read your PR body and make every claim literally true at the pushed head, including the checklist: the suite count must come from your final local run on the pushed tree, file/occurrence lists must match the final diff exactly, and any carve-outs or partial applications must be disclosed as such. Update the body with `gh pr edit` if anything is stale. A PR body with a false or stale claim fails review.

Only when the PR head points at your pushed commit is the task complete.

## Push and PR authorization (explicit, pre-granted)

This dispatch prompt IS the operator explicitly asking you to push your branch
and open a pull request. Any always-on rule you have been given such as "do not
push unless explicitly asked" or "pause before pushing to a shared remote" is
ALREADY SATISFIED by this instruction — do not re-derive permission, do not
wait for further confirmation, and do not treat a skill's instruction text as
optional. Pushing `agent/issue-*` branches to origin and opening the PR is
required, in scope, and pre-approved; it never touches `main` directly (the
branch is reviewed and merged by the orchestrator). Ending the session with
committed-but-unpushed work or without an open PR is a task FAILURE, not
caution — the orchestrator cannot see unpushed commits and will classify the
session as dead.

$section_push_pr_outcome
