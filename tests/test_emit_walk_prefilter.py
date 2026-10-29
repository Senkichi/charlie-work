"""Reachability control for the #2886 presence prefilter.

The per-file emit walk is the scanner's dominant cost, so
``_event_kind_scan`` skips any file whose text contains none of its presence
tokens; its soundness there is by construction (the tokens are derived from
the scanner's own flag tables plus the marker prefix), so this test is a
reachability control, not a re-derivation: a file whose ONLY trigger is a
flagged form must still be walked. Each parameter below pins one form on its
own, including the ``log_event (`` spelled-with-a-gap spelling a ``name(``
-shaped token match would silently drop.

This module keeps the pin out of ``test_event_kind_consumers.py`` itself, so
further scanner coverage can land without pushing that module back toward
its file-size ratchet mark -- the same hoisting convention ``_event_kind_scan``
follows (the zero-cross-test-import guard bans one ``test_*.py`` module
importing another).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _event_kind_scan import _scan_emit_sites


def _fixture_root_containing(tmp_path: Path, rel_path: str, body: str) -> Path:
    root = tmp_path / "src"
    parent = rel_path.rpartition("/")[0]
    if parent:
        package = root / parent
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    else:
        root.mkdir()
    (root / rel_path).write_text(body, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("rel_path", "body", "expected_func"),
    [
        (
            "file.py",
            'def go(path, payload):\n    log_event(path, "fixture_unconsumed_kind", payload)\n',
            "log_event",
        ),
        (
            "file.py",
            'def go(path, payload):\n    log_event (path, "fixture_unconsumed_kind", payload)\n',
            "log_event",
        ),
        (
            "file.py",
            'def go(path):\n    append_event(path, "fixture_unconsumed_kind", {})\n',
            "append_event",
        ),
        (
            "file.py",
            'def go():\n    _record_event({}, "fixture_unconsumed_kind")\n',
            "_record_event",
        ),
        (
            "file.py",
            'def go():\n    record_event({}, "fixture_unconsumed_kind")\n',
            "record_event",
        ),
        (
            "file.py",
            'def go(pr, a, b, c):\n    _route_to_rework(pr, a, b, c, "fixture_unconsumed_kind")\n',
            "_route_to_rework",
        ),
        (
            "file.py",
            'def flush():\n    sweep_events.append(("fixture_unconsumed_kind", {}))\n',
            "sweep_events.append",
        ),
        (
            "dead_worker_sweep/decide_fixture.py",
            'def decide():\n    return emit("fixture_unconsumed_kind", {})\n',
            "emit",
        ),
        (
            "dead_worker_sweep/decide_fixture.py",
            'def decide():\n    return StateTxn(event_kind="fixture_unconsumed_kind")\n',
            "event_kind",
        ),
        (
            "dead_worker_sweep/decide_fixture.py",
            'def decide():\n    return GuardedUpdate(event=("fixture_unconsumed_kind", {}))\n',
            "event_kind",
        ),
    ],
    ids=[
        "flagged_call",
        "flagged_call_name_paren_gap",
        "append_event",
        "record_event_underscored",
        "record_event_bare",
        "route_to_rework",
        "sweep_events_receiver",
        "decide_emit_call",
        "decide_event_kind_keyword",
        "decide_event_tuple_keyword",
    ],
)
def test_every_flagged_emit_form_reaches_the_presence_prefilter(
    tmp_path: Path, rel_path: str, body: str, expected_func: str
) -> None:
    """Each form ``_scan_emit_sites_in_tree`` can flag passes ``needs_emit_walk``."""
    root = _fixture_root_containing(tmp_path, rel_path, body)
    sites = [s for s in _scan_emit_sites(root) if s.func_name == expected_func]
    assert sites, f"{expected_func} form not reached by the walk (file: {rel_path})"
    assert len(sites) == 1, f"unexpected extra sites: {[(s.path, s.lineno) for s in sites]}"
    assert "fixture_unconsumed_kind" in sites[0].kind_source
