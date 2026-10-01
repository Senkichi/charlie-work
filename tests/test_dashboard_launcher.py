"""Shape guards for the charlie-dashboard scheduled-task assets."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def _task() -> ET.Element:
    return ET.parse(SCRIPTS / "charlie-dashboard-task.xml").getroot()


def test_task_triggers_at_logon_and_retries_on_failure() -> None:
    root = _task()
    assert root.find("t:Triggers/t:LogonTrigger", NS) is not None
    assert root.find("t:Settings/t:RestartOnFailure/t:Count", NS).text == "3"
    assert root.find("t:Settings/t:RestartOnFailure/t:Interval", NS).text == "PT1M"
    assert root.find("t:Settings/t:MultipleInstancesPolicy", NS).text == "IgnoreNew"
    assert root.find("t:RegistrationInfo/t:URI", NS).text == "\\charlie-dashboard"


def test_task_runs_hidden_vbs_wrapper() -> None:
    args = _task().find("t:Actions/t:Exec/t:Arguments", NS).text
    assert "dashboard-hidden.vbs" in args
    assert (SCRIPTS / "dashboard-hidden.vbs").is_file()


def test_vbs_waits_and_propagates_exit_code() -> None:
    vbs = (SCRIPTS / "dashboard-hidden.vbs").read_text(encoding="utf-8")
    assert '\\dashboard.ps1"""' in vbs
    assert ", 0, True)" in vbs
    assert "WScript.Quit exitCode" in vbs


def test_launcher_restart_code_matches_wire_contract() -> None:
    ps1 = (SCRIPTS / "dashboard.ps1").read_text(encoding="utf-8-sig")
    match = re.search(r"^\$ExitRestartRequested = (\d+)\s*$", ps1, re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) == EXIT_RESTART_REQUESTED


def test_launcher_shape() -> None:
    raw = (SCRIPTS / "dashboard.ps1").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "PS 5.1 needs a UTF-8 BOM"
    code = "\n".join(
        line for line in raw.decode("utf-8-sig").splitlines() if not line.lstrip().startswith("#")
    )
    assert "python -m charlie_work dashboard serve" in code  # not the console script
    assert "*>>" not in code and "2>&1" in code  # redirect inside cmd
    assert "PYTHONIOENCODING = 'utf-8:surrogateescape'" in code
    assert "exit $exitCode" in code
