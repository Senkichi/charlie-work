"""Reachability and skip control for the #2886 presence prefilter.

The per-file emit walk is the scanner's dominant cost, so ``_event_kind_scan``
skips any file whose text contains none of its presence tokens. Both halves of
the gate are pinned here, with direct assertions on the symbols under test
(``needs_emit_walk``, ``_PRESENCE_TOKENS`` and ``_scan_emit_sites`` against the
walk it is allowed to skip):

- Positive reachability (whose soundness in the skip direction is by
  construction -- the tokens are derived from the scanner's own flag tables
  plus the marker prefix -- so this half is a reachability control, not a
  re-derivation): a file whose ONLY trigger is a flagged form must still be
  walked. Each parameter below pins one form on its own, including the
  ``log_event (`` spelled-with-a-gap spelling a ``name(``-shaped token match
  would silently drop.
- Negative skip: a non-decide file whose text carries no presence token gets
  ``needs_emit_walk`` == False and never enters ``_scan_emit_sites_in_tree``
  through ``_scan_emit_sites`` (pinned by spying the walk through
  ``monkeypatch``), while ``MARKER_PREFIX``-bearing and decide files keep
  their walk even with zero sites.

This module keeps the pins out of ``test_event_kind_consumers.py`` itself, so
further scanner coverage can land without pushing that module back toward
its file-size ratchet mark -- the same hoisting convention ``_event_kind_scan``
follows (the zero-cross-test-import guard bans one ``test_*.py`` module
importing another).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _event_kind_scan
from _event_kind_scan import (
    ALL_EMIT_LIKE_NAMES,
    MARKER_PREFIX,
    SWEEP_APPEND_RECEIVER,
    _PRESENCE_TOKENS,
    _scan_emit_sites,
    _scan_emit_sites_in_tree,
    needs_emit_walk,
)


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


# ---------------------------------------------------------------------------
# The negative half of the gate: no presence token => no walk.
# ---------------------------------------------------------------------------

_TOKEN_FREE_BODY = "def plain():\n    return 1\n"


def test_needs_emit_walk_is_false_for_non_decide_file_without_any_presence_token() -> None:
    """A token-free ``file.py`` cannot hide a site, so the walk is skipped."""
    assert needs_emit_walk("file.py", _TOKEN_FREE_BODY) is False


@pytest.mark.parametrize("token", sorted(_PRESENCE_TOKENS), ids=lambda t: repr(t))
def test_every_presence_token_alone_satisfies_needs_emit_walk(token: str) -> None:
    """Each token in the set, on its own in the text, keeps the file walked."""
    assert token
    assert needs_emit_walk("file.py", token + "\n") is True


def test_presence_tokens_come_from_the_scanners_own_flag_tables() -> None:
    """The token set stays derived from the scanner's flags, never narrowed.

    Soundness of the skip rests on this: narrowing the set below the union of
    the emit-like names, the sweep receiver and the marker prefix would let a
    file whose only trigger is the dropped shape fall out of the walked set
    with real sites still inside.
    """
    assert _PRESENCE_TOKENS >= ALL_EMIT_LIKE_NAMES
    assert {SWEEP_APPEND_RECEIVER, MARKER_PREFIX} <= _PRESENCE_TOKENS


def test_needs_emit_walk_is_true_for_marker_file_and_decide_module() -> None:
    """Marker-bearing and decide files keep the walk even with zero sites."""
    marker_text = f"{MARKER_PREFIX} audit-only - review telemetry only\n"
    assert needs_emit_walk("file.py", marker_text) is True
    assert needs_emit_walk("dead_worker_sweep/decide_flagged.py", _TOKEN_FREE_BODY) is True


def test_scan_emit_sites_skips_the_walk_on_non_matching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_scan_emit_sites`` only hands matching files to the walk.

    The fixture tree holds one token-free file, one file whose only trigger is
    a flagged call, and one marker-bearing file with no emit call at all. The
    walk is wrapped with a recorder that calls the real implementation, so the
    assertion is direct on what ``_scan_emit_sites`` enters it with: only the
    matching and marker-bearing relpaths, never the token-free one.
    """
    root = _fixture_root_containing(
        tmp_path,
        "flagged.py",
        'def go(path, payload):\n    append_event(path, "fixture_unconsumed_kind", payload)\n',
    )
    (root / "plain.py").write_text(_TOKEN_FREE_BODY, encoding="utf-8")
    (root / "marked.py").write_text(
        f"{MARKER_PREFIX} audit-only - review telemetry only\n", encoding="utf-8"
    )

    walked: list[str] = []
    original = _scan_emit_sites_in_tree

    def recording_walk(tree, rel_path: str):
        walked.append(rel_path)
        return original(tree, rel_path)

    monkeypatch.setattr(_event_kind_scan, "_scan_emit_sites_in_tree", recording_walk)

    sites = _scan_emit_sites(root)

    assert walked == ["flagged.py", "marked.py"], (
        f"the walk ran on files the prefilter should skip: {walked}"
    )
    assert [s.func_name for s in sites] == ["append_event"]
