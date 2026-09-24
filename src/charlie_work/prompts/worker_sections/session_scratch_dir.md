## Scratch files — `$$TMPDIR` only, never a literal `/tmp/...` path

Every scratch file this session creates — captured command output,
downloaded blobs, throwaway scripts — MUST live under `$$TMPDIR`. The
orchestrator exports `TMP`/`TEMP`/`TMPDIR` pointed at a per-session
directory inside your worktree, so `mktemp` and `"$$TMPDIR/name"` already
land somewhere private to this session:

```bash
gh pr view 123 --json body -q .body > "$$TMPDIR/pr-body.md"
```

Never use a literal `/tmp/...` path in a shell command — not in a redirect
(`> /tmp/x`), a `mktemp` template, `cd`, or a tool argument. Under Git
Bash, a `/tmp/...` path resolves through an MSYS mount cached per
Git-for-Windows install — shared across every worker session on the
host — and on any platform it is a single directory outside your
worktree that nothing scopes to this session. A concurrent session can
read, overwrite, or delete a file there mid-use; exactly that shape
(`gh pr view ... > /tmp/pr-body.md`) corrupted a PR body in production.
Do not unset `TMPDIR`, and do not substitute a literal path "for
convenience" — the variable is already exported in your environment.
