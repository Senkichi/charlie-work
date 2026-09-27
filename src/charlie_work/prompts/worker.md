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
$attachment_budget
$section_scope_contract

$section_spec_letter

$section_no_merge_contract

$section_process_lifetime

$section_session_scratch_dir

$section_api_shape_validation

$section_caller_sweep

$section_parallel_investigation

$section_config_parity

$section_available_skills

## Required implementation loop

1. $section_loop_branch_step
2. Read `CLAUDE.md` (where present), `CONTRIBUTING.md` (where present),
   `CONTEXT.md` (where present), the issue, and relevant code.
3. Reproduce or precisely explain the defect/requirement.
4. Implement the smallest correct change.

   $section_invariant_enumeration
5. Run the tests impacted by your change from the worktree root: the test
   file(s) you added or modified, plus `grep tests/` for every
   module/function/symbol your production diff touched and run every
   matching test file — not just the tests you wrote:
   ```bash
   $targeted_test_command
   ```
   $section_execution_contract
   $section_loop_test_skill_note
   $section_repo_commands_precedence
6. Add or update regression tests unless not applicable.
7. $section_ruff_preflight
$section_loop_finish_steps

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

You are done once your branch is pushed, the push is verified, and you have
written `.worker-outcome.json` with your drafted PR title/body (including
`Closes #$issue_number` and a clear verification summary) per "Push and PR
outcome" below. The orchestrator, which is authenticated, opens the PR itself
from that file immediately after your session ends -- do not run `gh pr create`
or wait for a PR to appear before exiting.

**Committing locally is NOT done.** The branch must be pushed and the push verified.

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
3. After verifying the push, re-read your PR body and make every claim literally true at the pushed head, including the checklist: the suite count must come from your final local run on the pushed tree, file/occurrence lists must match the final diff exactly, and any carve-outs or partial applications must be disclosed as such. This is the body you write into `.worker-outcome.json` as `pr_body` -- there is no PR yet to run `gh pr edit` against, so get it right before writing the file. A PR body with a false or stale claim fails review.
4. Follow "Push and PR outcome" below to draft your PR title/body per the
   reconciliation above and write `.worker-outcome.json`, then stop.

Only when the pushed head matches your local HEAD and the outcome file is written is the task complete.

## Push authorization (explicit, pre-granted)

This dispatch prompt IS the operator explicitly asking you to push your branch.
Any always-on rule you have been given such as "do not push unless explicitly
asked" or "pause before pushing to a shared remote" is ALREADY SATISFIED by
this instruction — do not re-derive permission, do not wait for further
confirmation, and do not treat a skill's instruction text as optional. Pushing
`agent/issue-*` branches to origin is required, in scope, and pre-approved; it
never touches `main` directly (the branch is reviewed, PR-opened, and merged
by the orchestrator). Ending the session with committed-but-unpushed work, or
without a written `.worker-outcome.json`, is a task FAILURE, not caution — the
orchestrator cannot see unpushed commits and will classify the session as dead.

$section_push_pr_outcome
