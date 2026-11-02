' Windowless launcher for dashboard.ps1 (charlie-dashboard scheduled task).
'
' Same pattern as fleet-pass-hidden.vbs: wscript.exe is a GUI-subsystem host and
' Run(..., 0, True) starts powershell with SW_HIDE, so no console ever shows.
' The .ps1 path is derived from this script's own location -- keep both files
' in the same directory.
'
' bWaitOnReturn MUST stay True: the task tracks wscript.exe, so waiting keeps the
' task "Running" for the life of the server (MultipleInstancesPolicy=IgnoreNew).
' The exit code MUST be propagated via WScript.Quit: Task Scheduler's
' restart-on-failure policy only sees a non-zero result if wscript returns it.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
exitCode = shell.Run("powershell.exe -NonInteractive -NoProfile -ExecutionPolicy Bypass -File """ & scriptDir & "\dashboard.ps1""", 0, True)
WScript.Quit exitCode
