## Scope contract

- Solve only issue #$issue_number.
- Do not batch unrelated fixes.
- Do not perform opportunistic refactors.
- Preserve the patterns and invariants in `CLAUDE.md`.
- If the issue is ambiguous, stop and explain the blocker instead of guessing.
- If the fix touches security-sensitive behavior, call it out explicitly in the PR.
- **Containment:** All file edits, git operations, and test runs happen in the session's current working directory (the assigned worktree); never resolve, cd into, or modify any other checkout of the repo (including the parent working tree a git worktree points back to).