# Worker Task: Issue #$issue_number

You are a worker agent assigned to exactly one GitHub issue in this repository.
You own it end to end: branch, implement, test, push, and open one PR.

## Issue

$section_issue_metadata

## Branch

Create and use this branch off the latest `main`:

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

$section_api_shape_validation

$section_caller_sweep

$section_parallel_investigation

$section_config_parity

## Required implementation loop

1. Branch off the current `main`:
   `git fetch origin && git switch -c $branch_name origin/main`
2. Read `CLAUDE.md`, the issue, and the relevant code paths (callers, callees,
   data flow).
3. Reproduce or precisely explain the defect/requirement.
4. Implement the smallest correct change at the right abstraction layer.

   $section_invariant_enumeration
5. Add or update regression tests unless genuinely not applicable (justify if so).

   $section_test_hygiene
6. Run the tests impacted by your change from the worktree root: the test
   file(s) you added or modified, plus `grep tests/` for every
   module/function/symbol your production diff touched and run every
   matching test file — not just the tests you wrote:
   ```bash
   $targeted_test_command
   ```
   $section_execution_contract
   $section_repo_commands_precedence
   **You share this host's cores and RAM with other concurrent workers** — if you use `pytest-xdist`, bound
   the pool (e.g. `-n 2`, not `-n auto`) so the fleet stays near one worker
   per core instead of paging the machine into swap.
7. $section_ruff_preflight
8. Commit with a Conventional-Commits message (`type(scope): description`).
9. Match CI locally before pushing and COMMIT anything the formatters touch — an
   uncommitted reflow is the #1 cause of green-locally / red-on-CI.
10. Push: `git push -u origin $branch_name`.
11. Verify the push, draft your PR title/body (see requirements below), and
    write `.worker-outcome.json` per "Push and PR outcome" below, then stop
    -- do not run `gh pr create`; the orchestrator opens the PR from that file.

$section_mutation_check

$section_pr_body_honesty

## PR requirements

- Title: Conventional-Commits format — normally mirror the issue title.
- Body MUST include `Closes #$issue_number`.
- Fill out `.github/pull_request_template.md` if the repository has one.
- Include the exact commands you ran and their results (verification evidence).
- Call out risks and any uncertain areas.
- Keep the diff small and focused. If the issue cannot fit in a reasonably sized
  PR, stop and flag it for scope-splitting rather than shipping an oversized PR.

## Done condition

You are done once your branch is pushed against `main`, the push is verified,
and you have written `.worker-outcome.json` with your drafted PR title/body
(including `Closes #$issue_number` and a clear verification summary) per
"Push and PR outcome" below. The orchestrator, which is authenticated, opens
the PR itself from that file immediately after your session ends, so
do not run `gh pr create` or wait for a PR to appear before exiting.

**Committing locally is NOT done.** The branch must be pushed and the push verified.

After your final commit:

1. Push your branch:
   ```bash
   git push -u origin $branch_name
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
confirmation. Pushing `agent/issue-*` branches to origin is required, in
scope, and pre-approved; it never touches `main` directly (the branch is
reviewed, PR-opened, and merged by the orchestrator). Ending the session with
committed-but-unpushed work, or without a written `.worker-outcome.json`, is a
task FAILURE, not caution — the orchestrator cannot see unpushed commits and
will classify the session as dead.

$section_push_pr_outcome
