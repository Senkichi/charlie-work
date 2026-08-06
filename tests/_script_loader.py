"""Shared helper to load a scripts/ file as a module in tests.

The duplication this removes is the "load a Python file as a module" recipe
that lived in six test files and drifted once already (issue #1023). Central
imports and usage:

* tests/test_heartbeat_check.py
* tests/test_backfill_stale_rework_briefs.py
* tests/test_ac1b_findings_actionability.py
* tests/test_doctor.py
* tests/test_verify_events.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script_module(
    script_path: Path,
    module_name: str,
    argv: list[str] | None = None,
) -> ModuleType:
    """Load ``script_path`` as ``module_name`` without adding it to ``sys.path``.

    The module is registered in ``sys.modules`` *before* ``exec_module`` so
    scripts that use ``from __future__ import annotations`` can resolve their
    string annotations during class creation (issue #1023). Any previous entry
    for ``module_name`` and the previous ``sys.argv`` value are saved and
    restored in ``finally``, which makes the helper safe both for module-scoped
    fixtures (the module is loaded once and the same object is reused) and for
    per-call reloads (the module is re-executed fresh each time under the same
    name).
    """
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {script_path}")

    module = importlib.util.module_from_spec(spec)

    prior_module = sys.modules.get(module_name)
    sys.modules[module_name] = module

    prior_argv: list[str] | None = None
    if argv is not None:
        prior_argv = sys.argv
        sys.argv = argv

    try:
        spec.loader.exec_module(module)
    finally:
        if prior_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_module
        if prior_argv is not None:
            sys.argv = prior_argv

    return module
