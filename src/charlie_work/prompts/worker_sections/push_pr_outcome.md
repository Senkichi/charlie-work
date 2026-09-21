## Push and PR outcome

Do not run `gh pr create`, `gh pr view`, or look for any other way to open the pull request yourself (including an MCP GitHub server, if one appears to be configured). Workers carry no `gh` credential by design (issue #502): the attempt always fails on missing authentication and only burns tool calls on a guaranteed dead end. The orchestrator is authenticated and opens every PR itself, immediately after your session ends.

Once you have:

1. committed your changes,
2. pushed the branch (`git push -u origin $branch_name` or equivalent), and
3. verified the push -- the first column of `git ls-remote origin $branch_name` equals `git rev-parse HEAD`,

write a file named `.worker-outcome.json` in the repository root with this exact shape:

```json
{"push_succeeded": true, "pr_created": false, "pr_title": "<Conventional-Commits PR title>", "pr_body": "<full PR body>"}
```

`pr_title` and `pr_body` are the real title and body you would have used for `gh pr create` -- draft them completely, per the PR requirements above (title format, `Closes #$issue_number`, the exact commands you ran and their results, risks and uncertain areas). The orchestrator uses this text verbatim when it opens the PR, so do not leave either field blank, a placeholder, or a promise to fill it in later.

Then stop. Writing this file, with the push already verified, is the task's done condition -- do not wait for a PR to appear, and do not treat the absence of an open PR as a reason to keep working.

Only write this file when the push itself succeeded and was verified. If the push failed, do not write this file; report the push failure in your session log instead.

$section_blocked_outcome
