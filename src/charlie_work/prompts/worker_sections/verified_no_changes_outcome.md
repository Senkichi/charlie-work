## Worker-declared verified-no-changes outcome

Some tickets are verification-only: their spec says no code change is expected ("Files: Modify: (none expected -- verification and fixup only)") and tells you to commit only if fixes were needed. If you ran the verification, everything passed, and you have **no commits and no source changes**, do not push and do not open a PR. Write a file named `.worker-outcome.json` in the repository root with this exact shape:

```json
{"outcome": "verified_no_changes", "detail": "<evidence: the commands you ran and their results>"}
```

Then exit cleanly. The orchestrator closes the issue with `detail` recorded as its resolution -- no redispatch, no cap burn. It honours the outcome only when your worktree has no commits ahead of the base and no uncommitted source changes, and the issue is not escalated; otherwise the file is ignored and the session is handled as an ordinary dead worker. If you made any fix, commit and push it and use the normal PR outcome above instead -- never use this outcome to abandon unfinished work.
