# Worker Task: Issue #$issue_number

You are a worker agent assigned to exactly one GitHub issue in this repository.
You own it end to end: branch, implement, test, push, and open one PR.

## Issue

- Number: #$issue_number
- Title: $issue_title
- URL: $issue_url
- Model tier target: $worker_model_tier

## Branch

Create and use this branch off the latest `main`:

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

## Required implementation loop

1. Branch off the current `main`:
   `git fetch origin && git switch -c $branch_name origin/main`
2. Read `CLAUDE.md`, the issue, and the relevant code paths (callers, callees,
   data flow).
3. Reproduce or precisely explain the defect/requirement.
4. Implement the smallest correct change at the right abstraction layer.
5. Add or update regression tests unless genuinely not applicable (justify if so).
6. Run the test suite with the repository's canonical test command (see `CLAUDE.md`).
7. Match CI locally before pushing and COMMIT anything the formatters touch — an
   uncommitted reflow is the #1 cause of green-locally / red-on-CI.
8. Commit with a Conventional-Commits message (`type(scope): description`).
9. Push: `git push -u origin $branch_name`.
10. Open the PR (see requirements below).

## PR requirements

- Title: Conventional-Commits format — normally mirror the issue title.
- Body MUST include `Closes #$issue_number`.
- Fill out `.github/pull_request_template.md` if the repository has one.
- Include the exact commands you ran and their results (verification evidence).
- Call out risks and any uncertain areas.
- Keep the diff small and focused. If the issue cannot fit in a reasonably sized
  PR, stop and flag it for scope-splitting rather than shipping an oversized PR.

## Done condition

You are done only when the PR is open against `main`, linked to issue
#$issue_number via `Closes #$issue_number`, CI has been given a clean tree, and
the PR body contains a clear verification summary.
