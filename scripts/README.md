# scripts/

Operator utilities for running and observing the charlie-work fleet
supervisor on this host. These are host-side helpers, not part of the
`charlie` package — several of them intentionally hardcode paths or stay
stdlib-only; see the notes below before "fixing" one.

## Files

- **`fleet-pass.ps1`** — launches `charlie fleet supervise` (the long-running
  daemon), re-invoked every 5 minutes by the `charlie-fleet-pass` scheduled
  task as a crash-restart watchdog. Repo root is derived from the script's
  own location (`$PSScriptRoot`); the log directory
  (`.var\charlie-work\logs`) is a literal that mirrors the default
  `runtime.state_dir` (`config.RuntimeConfig`) and must be updated by hand if
  a repo's config ever overrides that setting. Lines implementing the PS 5.1
  `cmd`-redirect workaround (native stderr reordering/dropping under
  PowerShell — see the inline comments) are load-bearing; do not touch them
  when editing this file.
- **`fleet-pass-hidden.vbs`** — windowless launcher for `fleet-pass.ps1`,
  invoked by the scheduled task via `wscript.exe` so no console window
  flashes. Already self-locating (derives `fleet-pass.ps1`'s path from its
  own `WScript.ScriptFullName`); keep both files in the same directory.
- **`charlie-fleet-pass-task.xml`** — the Windows Task Scheduler definition
  for `charlie-fleet-pass`, exported via `schtasks /query /xml`. **It contains
  absolute paths** (`C:\Users\senki\repos\charlie-work\scripts\...`) that are
  baked into the *registered* task, not read from this file at run time.
  Editing this XML does **not** update the live task — if the repo is ever
  moved or checked out under a different user profile, re-import it with
  `schtasks /create /tn charlie-fleet-pass /xml charlie-fleet-pass-task.xml`
  (or `Register-ScheduledTask`) after fixing the paths inside.
- **`monitor_events.py`** / **`verify_events.py`** — read `events.db`
  (via `charlie_work.instrumentation`) for live monitoring / one-shot health
  checks. Both accept an optional `state.json` path as `argv[1]`; when
  omitted, the path is resolved from the current repo's layered config
  (`runtime.state_dir`) via `charlie_work.global_config.load_layered_config`,
  so they target the right tree on any repo that overrides the default
  `.var/charlie-work`.
- **`heartbeat_check.py`** — deterministic fleet-heartbeat check (see its
  module docstring). Deliberately **stdlib-only** (plus `psutil`/`yaml`,
  already project dependencies) so a broken package install can never break
  the check that would detect it. Do not add a `charlie_work` import here.
- **`backfill_stale_rework_briefs.py`** — one-shot operator tool (F6 of
  `docs/plans/rework-findings-channel.md`) that bumps a `review-decision.json`
  verdict's mtime (`os.utime` only — never rewrites its contents) for PRs
  whose existing `rework-prompt.md` brief will otherwise be reused verbatim
  forever, so the next `dispatch_rework` pass regenerates the brief through a
  fixed renderer. Dry-run by default; `--apply` refuses to run unless
  `--require-commit` proves the renderer fix is an ancestor of the target
  repo's live HEAD (not bypassable). See the module docstring before use —
  applying before the fix is deployed silently burns the backfill's only
  lever. Accepts `--repo` to target a different checkout's `.var/` state
  (e.g. running from an isolated worktree against the main checkout).
