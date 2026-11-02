# Fleet dashboard

Read-only, derived view of the fleet (see ADR-0008). It binds 127.0.0.1 only and never
writes fleet state or calls GitHub.

## Running the dashboard

Foreground, for a quick look:

```powershell
uv run charlie dashboard serve
```

As a logon-started scheduled task (hidden window, log at
`.var\charlie-work\logs\dashboard.log`). Files: `scripts/charlie-dashboard-task.xml`,
`scripts/dashboard-hidden.vbs`, `scripts/dashboard.ps1`. The launcher relaunches the
server while it exits with `EXIT_RESTART_REQUESTED` (3, orchestrator HEAD drift); any
other non-zero exit is retried by Task Scheduler (3 retries, 1 minute apart).

Register once (fills in your SID and checkout path, then registers the task):

```powershell
$repo = 'C:\Users\senki\repos\charlie-work'
$sid  = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
$xml  = (Get-Content "$repo\scripts\charlie-dashboard-task.xml" -Raw -Encoding UTF8).
    Replace('S-1-5-21-REPLACE-WITH-YOUR-USER-SID', $sid).
    Replace('C:\Users\YOUR_USERNAME\repos\charlie-work', $repo)
Register-ScheduledTask -TaskName 'charlie-dashboard' -Xml $xml
Start-ScheduledTask -TaskName 'charlie-dashboard'
```

Then open <http://127.0.0.1:8765/> (host and port come from the `dashboard` config
section; a non-loopback host is rejected as a config error).
