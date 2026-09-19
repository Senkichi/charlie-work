"""Layered config loading (``load_layered_config`` ``require_global``
semantics) and ``load_state`` corrupt-file quarantine.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from charlie_work.config import load_config


def test_load_state_quarantines_corrupt_file(tmp_path: Path) -> None:
    from charlie_work.state import load_state as _load

    state_path = tmp_path / "state.json"
    state_path.write_text("{truncated garbage", encoding="utf-8")

    state = _load(state_path)

    assert state["issues"] == {}
    assert not state_path.exists()
    quarantined = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "truncated garbage" in quarantined[0].read_text(encoding="utf-8")


def test_load_layered_config_require_global_raises_on_absent(tmp_path: Path) -> None:
    """A fleet entry point must not silently run on defaults when its global
    layer is missing.

    The default ``require_global=False`` path is legitimate for a plain
    single-repo checkout (no global layer, no fleet-wide knobs expected). But a
    fleet supervisor that loads the global layer with ``require_global=True``
    and gets nothing back has lost every host-wide knob -- ``runner_allocation``
    off, ``notify`` off, ``labels`` back to built-ins -- while passes keep
    reporting success. That is #590's exact symptom and #623's root cause: the
    resolved config records what a section *became*, never whether the file that
    declares it was *read*. The raise makes the read-failure loud instead of
    byte-identical with "the operator left it off".

    The error must name the path (so the operator knows *which* config.yaml was
    expected) and the ``describe_config_file`` cause (so "never created" is
    distinguishable from "the volume was not ready" -- ``exists()`` collapses
    both into a bare ``False`` and this is the one place that un-collapses them).
    """
    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # No config.yaml in the fleet dir -- the silent-{} branch.
    with pytest.raises(ConfigError) as exc_info:
        load_layered_config(
            repo_root,
            None,
            fleet_dir_override=str(fleet_dir_path),
            require_global=True,
        )
    message = str(exc_info.value)
    assert str(fleet_dir_path / "config.yaml") in message, "the expected path must be named"
    assert "absent" in message, (
        f"an absent global layer must read as absent, not collapse to a bare failure: {message!r}"
    )

    # The default (single-repo) path is unchanged: no raise, pristine defaults.
    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    default_config = load_config(None)
    assert config.labels.ready == default_config.labels.ready


def test_load_layered_config_require_global_raises_on_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable global layer must not read as absent under require_global.

    ``Path.exists()`` swallows every error in ``pathlib._ignore_error`` --
    ENOENT, ENOTDIR, EBADF, ELOOP and the Windows device-not-ready /
    unresolvable-path winerrors -- into a bare ``False``. So an unready volume
    and a file that was never created take the same silent-``{}`` branch and
    produce byte-identical pristine defaults. The ``ConfigError`` raised under
    ``require_global=True`` must carry the ``describe_config_file`` cause so the
    two are separable in the error message -- they demand opposite fixes.

    ENOTDIR is used rather than EACCES deliberately: permission errors are *not*
    in the ignored set, so they raise from ``exists()`` before reaching the
    ``require_global`` check and are therefore not a silent-default mechanism.
    """
    import errno

    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("dispatch: {}\n", encoding="utf-8")

    real_stat = Path.stat

    def not_a_dir(self: Path, *args: object, **kwargs: object) -> object:
        if self == global_config_path:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", not_a_dir)

    # Precondition: exists() collapses ENOTDIR to False, which is the silent
    # default path the raise is meant to replace.
    assert global_config_path.exists() is False

    with pytest.raises(ConfigError) as exc_info:
        load_layered_config(
            repo_root,
            None,
            fleet_dir_override=str(fleet_dir_path),
            require_global=True,
        )
    message = str(exc_info.value)
    assert "UNREADABLE" in message, (
        f"an unreachable global layer must not read as absent: {message!r}"
    )
    assert "NotADirectoryError" in message, "the cause must survive into the error"
    assert "absent" not in message, "an unreachable path is not absent"


def test_load_layered_config_require_global_present_loads(tmp_path: Path) -> None:
    """A present, readable global layer must load normally under require_global.

    ``require_global`` only changes the absent/unreachable branch; a present
    file with real content must behave identically to the default path. This
    guards against the fix accidentally making the happy path raise.
    """
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    (fleet_dir_path / "config.yaml").write_text(
        "dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8"
    )

    config = load_layered_config(
        repo_root,
        None,
        fleet_dir_override=str(fleet_dir_path),
        require_global=True,
    )
    assert config.dispatch.max_concurrent_sessions == 5


def test_load_layered_config_require_global_present_but_invalid_keeps_per_repo(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A present-but-invalid global layer under require_global must not discard
    the per-repo config.

    Regression for the second orchestrator review of PR #630: the caller-level
    ``require_global=True`` fallback (catch ``ConfigError``, reload with
    ``require_global=False``) must not be the thing that rescues a valid
    per-repo config from a broken global layer -- that rescue happens *inside*
    ``load_layered_config`` itself (the in-function fallback added by #667), so
    the caller's catch block never fires for this case. This test proves the
    per-repo config survives a present-but-invalid global layer *without* the
    caller-level retry, i.e. ``require_global=True`` returns successfully and
    the per-repo value is intact.
    """
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Global layer is present but invalid (unknown top-level section).
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("unknown_section:\n  foo: bar\n", encoding="utf-8")

    # Per-repo config is valid and carries a distinctive value.
    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text("dispatch:\n  max_concurrent_sessions: 7\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.global_config"):
        config = load_layered_config(
            repo_root,
            None,
            fleet_dir_override=str(fleet_dir_path),
            require_global=True,
        )

    # The per-repo value survives -- the broken global layer did not discard it.
    assert config.dispatch.max_concurrent_sessions == 7

    # The discard is loud, not silent: a warning names the discarded global path.
    warning = "\n".join(
        r.getMessage() for r in caplog.records if "global layer was discarded" in r.getMessage()
    )
    assert warning, "discarding an invalid global layer must be logged as a warning"
    assert str(global_config_path) in warning, "the warning must name the global path"


def test_layered_config_logs_whether_global_layer_was_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The provenance line must report *readership*, not resolved values.

    A section that is missing from the global file and a global file that was
    never read produce byte-identical resolved config -- pristine dataclass
    defaults -- but demand opposite fixes. Issue #590 stalled on exactly that
    ambiguity, so this asserts the one fact that separates them survives in the
    log even when the resulting config looks entirely ordinary.
    """
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Absent global layer: resolved config is all defaults, and the log says why.
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    absent_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "absent" in absent_line, f"absent global layer not reported: {absent_line!r}"
    assert str(fleet_dir_path / "config.yaml") in absent_line, "the path itself must be logged"

    # Present global layer: same call, and the log distinguishes it.
    caplog.clear()
    (fleet_dir_path / "config.yaml").write_text(
        "dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8"
    )
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    present_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "present" in present_line, f"present global layer not reported: {present_line!r}"
    assert "absent" not in present_line, "a present layer must not read as absent"
    assert "dispatch" in present_line, "the sections actually read must be named"
    assert "bytes=" in present_line, "same vocabulary as the supervisor's INFO line"

    # An empty global file resolves to pristine defaults with zero sections --
    # indistinguishable from an absent one on the resolved config alone, and the
    # reason this line carries a byte count rather than a bare "present".
    caplog.clear()
    (fleet_dir_path / "config.yaml").write_text("", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    empty_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "bytes=0" in empty_line, f"a truncated global layer must be visible: {empty_line!r}"
    assert "absent" not in empty_line, "an empty file is present-but-empty, not absent"
    assert "(none)" in empty_line, "an empty file contributes no sections"
