## Process lifetime

Do not launch detached, disowned, or backgrounded long-running processes: `nohup`, `setsid`, `start /b`, `Start-Process` (PowerShell), a trailing `&`, a scheduled task, or a cron entry. Everything you start must complete, or be killed by you, within the foreground lifetime of this session. A long-running drain, backfill, or server is orchestrator-side work, not a worker's — a detached process can silently mutate shared state after the session that spawned it has ended, with nobody watching it.

If a long-running process is genuinely required, redirect its stdout and stderr to a file inside your worktree and verify genuine progress (not just that it started) within the first 5 minutes. If there is no progress signal at 5 minutes, kill it and diagnose the cause rather than waiting on a process that has gone silent.
