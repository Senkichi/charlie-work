' Windowless launcher for fleet-pass.ps1 (charlie-fleet-pass scheduled task).
'
' Why this exists: the task previously ran powershell.exe directly with
' -WindowStyle Hidden. For console-subsystem executables Windows creates the
' console window BEFORE the process can act on -WindowStyle Hidden, so every
' 5-minute pass flashed a visible terminal for the duration of PowerShell's
' startup. wscript.exe is a GUI-subsystem host (never gets a console), and
' Run(..., 0, False) starts the child with SW_HIDE in its STARTUPINFO — the
' console is created already-hidden and is never shown. Child processes of
' the pass (uv/python/gh/git) attach to that same hidden console, so they
' cannot flash either.
'
' The .ps1 path is derived from this script's own location — keep both files
' in the same directory.
'
' bWaitOnReturn MUST stay True: the scheduled task tracks wscript.exe, so
' waiting keeps the task "Running" for the life of the daemon — preserving
' MultipleInstancesPolicy=IgnoreNew (no overlapping launches while the
' supervisor is alive). With False the task would "complete" instantly and
' IgnoreNew would stop suppressing double-launches.
'
' ExecutionTimeLimit is PT0S (no limit) BY DESIGN and is NOT a guard here.
' The task launches `charlie fleet supervise --max-runtime 0` — a long-lived
' daemon (see fleet-pass.ps1). A real time limit would kill a healthy
' supervisor on a fixed timer. The one-shot era (re-invoked every 5 minutes)
' relied on ExecutionTimeLimit to bound a single pass; the daemon model does
' not. Do not "restore" a limit — it would terminate a healthy supervisor.
'
' The 5-minute trigger's watchdog role is crash recovery only: if the daemon
' process dies, no instance remains, IgnoreNew stops suppressing, and the
' next tick relaunches. A WEDGED daemon (alive but not making progress) is
' NOT detected by this trigger — IgnoreNew keeps suppressing while the
' process exists. Wedge detection is handled in-process by the supervise-loop
' wrapper's WedgeWatchdog (issue #728), which monitors the supervisor
' heartbeat and terminates a wedged child so the next tick relaunches it.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.Run "powershell.exe -NonInteractive -NoProfile -ExecutionPolicy Bypass -File """ & scriptDir & "\fleet-pass.ps1""", 0, True
