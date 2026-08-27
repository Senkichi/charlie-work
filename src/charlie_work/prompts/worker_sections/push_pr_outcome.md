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

## Worker-declared blocked outcome

If you deliberately conclude that you **cannot** do the task (the issue is structurally impossible for a contained worker — e.g. the fix belongs in a different repo, a required dependency is missing, or the issue scope is ambiguous beyond what you can resolve), write a file named `.worker-outcome.json` in the repository root with this exact shape:

```json
{"outcome": "blocked", "reason_kind": "cross_repo_scope", "detail": "The fix targets job-cannon, not this repo. The component lives in src/job_cannon/foo.py."}
```

Then exit cleanly without pushing a branch or opening a PR. The orchestrator reads this file on the first orphan-sweep pass and routes the issue directly to the operator queue — no redispatch, no cap burn. Another worker will not be dispatched for the same structural wall.

`reason_kind` must be one of:

- `cross_repo_scope` — the fix belongs in a different repo than the one you are contained in.
- `missing_dependency` — a required dependency, tool, or artifact is absent and cannot be installed or created within the worktree.
- `ambiguous_scope` — the issue's requirements are ambiguous beyond what you can resolve, and a reasonable implementation cannot be determined.
- `other` — none of the above; explain in `detail`.

`detail` must be a concise, actionable explanation of the blocker so the operator can re-scope or resolve it without reading your session log. Do **not** invent ad-hoc `BLOCKED.md` files — `.worker-outcome.json` is the channel the orchestrator consumes.
