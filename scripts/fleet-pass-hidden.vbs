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
' waiting keeps the task "Running" for the life of the pass — preserving
' ExecutionTimeLimit (hung-pass kill) and MultipleInstancesPolicy=IgnoreNew
' (no overlapping passes). With False the task would "complete" instantly
' and both guards would silently stop applying to the real pass process.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.Run "powershell.exe -NonInteractive -NoProfile -ExecutionPolicy Bypass -File """ & scriptDir & "\fleet-pass.ps1""", 0, True
