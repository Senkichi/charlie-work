## Push and PR outcome recovery

If the `gh` CLI is not authenticated and `gh pr create` fails **after** you have successfully pushed the branch `$branch_name` to `origin`, write a file named `.worker-outcome.json` in the repository root with this exact shape:

```json
{"push_succeeded": true, "pr_created": false, "error": "gh unauthenticated"}
```

Then exit cleanly. The orchestrator will read this file and open the PR itself for issue #$issue_number.

Only write this file when both of these are true:

- `git push -u origin $branch_name` (or equivalent) reported success, and
- `gh pr create` (or `gh pr view $branch_name`) failed due to missing `gh` authentication.

If the push itself failed, do **not** write this file; report the push failure in your session log instead.
