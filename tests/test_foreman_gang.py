"""``charlie gang new|list|show`` (issue #3117)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from charlie_work.cli import build_parser
from charlie_work.gang_command import run_gang_command
from charlie_work.foreman.store import scaffold_foreman


def _args(fleet: Path, command: str, name: str | None = None, from_dir: str | None = None):
    return argparse.Namespace(
        fleet_dir=str(fleet), gang_command=command, name=name, from_dir=from_dir
    )


def test_parser_accepts_gang_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["gang", "new", "my-gang"])
    assert args.command == "gang"
    assert args.gang_command == "new"
    assert args.name == "my-gang"
    args = parser.parse_args(["gang", "list"])
    assert args.gang_command == "list"
    args = parser.parse_args(["gang", "show", "my-gang"])
    assert args.name == "my-gang"


def test_gang_new_scaffolds_layout(tmp_path: Path) -> None:
    result = run_gang_command(_args(tmp_path, "new", "test-gang"))
    assert result.ok, result.message
    root = tmp_path / "foremen" / "test-gang"
    assert (root / "charter.yaml").is_file()
    assert (root / "instructions.md").is_file()
    assert (root / "memory").is_dir()
    assert (root / "turns").is_dir()
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert state["state"] == "paused"


def test_gang_new_rejects_bad_name_and_existing(tmp_path: Path) -> None:
    assert not run_gang_command(_args(tmp_path, "new", "Bad_Name")).ok
    assert run_gang_command(_args(tmp_path, "new", "g1")).ok
    result = run_gang_command(_args(tmp_path, "new", "g1"))
    assert not result.ok
    assert "already exists" in result.message


def test_gang_new_from_copies_documents(tmp_path: Path) -> None:
    source = tmp_path / "seed"
    source.mkdir()
    (source / "charter.yaml").write_text("name: g1\noutcome: seed\nrepos: []\n")
    (source / "instructions.md").write_text("seeded\n")
    result = run_gang_command(_args(tmp_path, "new", "g1", from_dir=str(source)))
    assert result.ok, result.message
    root = tmp_path / "foremen" / "g1"
    assert (root / "charter.yaml").read_text() == "name: g1\noutcome: seed\nrepos: []\n"
    assert (root / "instructions.md").read_text() == "seeded\n"
    # --from with missing documents is a clean failure.
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_gang_command(_args(tmp_path, "new", "g2", from_dir=str(empty)))
    assert not result.ok


def test_gang_new_from_undecodable_seed_is_a_clean_error(tmp_path: Path) -> None:
    """A UTF-16 seed decodes BEFORE any mkdir — an undecodable charter comes
    back as an error naming the file, not a traceback, and leaves no
    charterless ``foremen/<name>/`` behind to block retry."""
    source = tmp_path / "seed"
    source.mkdir()
    (source / "charter.yaml").write_bytes("name: g1\noutcome: seed\n".encode("utf-16"))
    (source / "instructions.md").write_text("seeded\n", encoding="utf-8")
    result = run_gang_command(_args(tmp_path, "new", "g1", from_dir=str(source)))
    assert not result.ok
    assert "charter.yaml" in result.message
    assert not (tmp_path / "foremen" / "g1").exists()


def test_gang_list_empty_is_clean_noop(tmp_path: Path) -> None:
    result = run_gang_command(_args(tmp_path, "list"))
    assert result.ok
    assert result.data["foremen"] == []
    assert "no foremen" in result.message


def test_gang_list_shows_state_and_bounds(tmp_path: Path) -> None:
    assert scaffold_foreman(tmp_path, "g1").ok
    (tmp_path / "fleet.json").write_text(json.dumps({"version": 1, "repos": {"acme/w": {}}}))
    (tmp_path / "foremen" / "g1" / "charter.yaml").write_text(
        'name: g1\noutcome: do it\nrepos: ["acme/w"]\nbounds:\n  max_open_issues: 9\n'
    )
    result = run_gang_command(_args(tmp_path, "list"))
    assert result.ok
    entry = result.data["foremen"][0]
    assert entry["name"] == "g1"
    assert entry["state"] == "paused"
    assert entry["display_state"] == "paused"
    assert entry["issues"] == "0/9"
    assert entry["turns_today"] == "0/12"


def test_gang_list_surfaces_invalid_charter_without_crashing(tmp_path: Path) -> None:
    assert scaffold_foreman(tmp_path, "g1").ok
    # The scaffold charter's placeholder repo is not registered — the foreman
    # still lists, with the charter error attached.
    result = run_gang_command(_args(tmp_path, "list"))
    assert result.ok
    entry = result.data["foremen"][0]
    assert "charter_error" in entry
    assert "charter invalid" in result.message


def test_gang_list_degraded_displays_as_cricket(tmp_path: Path) -> None:
    assert scaffold_foreman(tmp_path, "g1").ok
    (tmp_path / "foremen" / "g1" / "state.json").write_text(
        json.dumps({"version": 1, "state": "degraded"})
    )
    result = run_gang_command(_args(tmp_path, "list"))
    entry = result.data["foremen"][0]
    assert entry["state"] == "degraded"
    assert entry["display_state"] == "CRICKET"
    assert "CRICKET" in result.message


def test_gang_list_survives_non_utf8_charter(tmp_path: Path) -> None:
    """A non-UTF-8 ``charter.yaml`` surfaces as a per-foreman error — it cannot
    raise out of the list and hide the healthy foremen beside it."""
    assert scaffold_foreman(tmp_path, "bad").ok
    (tmp_path / "foremen" / "bad" / "charter.yaml").write_bytes("name: bad\n".encode("utf-16"))
    (tmp_path / "foremen" / "bad" / "state.json").write_bytes(
        '{"state": "active"}'.encode("utf-16")
    )
    assert scaffold_foreman(tmp_path, "good").ok
    (tmp_path / "fleet.json").write_text(json.dumps({"version": 1, "repos": {"acme/w": {}}}))
    (tmp_path / "foremen" / "good" / "charter.yaml").write_text(
        'name: good\noutcome: do it\nrepos: ["acme/w"]\n'
    )
    result = run_gang_command(_args(tmp_path, "list"))
    assert result.ok, result.message
    entries = {e["name"]: e for e in result.data["foremen"]}
    assert set(entries) == {"bad", "good"}
    assert "charter_error" in entries["bad"]
    assert entries["good"].get("issues") == "0/5"


def test_main_dispatches_gang_list(tmp_path: Path, capsys) -> None:
    """``charlie --fleet-dir <tmp> gang list`` reaches run_gang_command through
    ``cli.main`` — deleting the GANG_COMMAND dispatch branch fails this test."""
    from charlie_work import cli

    assert scaffold_foreman(tmp_path, "g1").ok
    rc = cli.main(["--fleet-dir", str(tmp_path), "gang", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "g1" in out


def test_gang_show(tmp_path: Path) -> None:
    assert scaffold_foreman(tmp_path, "g1").ok
    (tmp_path / "fleet.json").write_text(json.dumps({"version": 1, "repos": {"acme/w": {}}}))
    (tmp_path / "foremen" / "g1" / "charter.yaml").write_text(
        'name: g1\noutcome: do it\nrepos: ["acme/w"]\nstanding: true\narming: propose\n'
    )
    result = run_gang_command(_args(tmp_path, "show", "g1"))
    assert result.ok, result.message
    assert result.data["charter"]["standing"] is True
    assert result.data["charter"]["arming"] == "propose"
    assert result.data["state"]["state"] == "paused"
    assert "do it" in result.message


def test_gang_show_missing_is_a_clean_error(tmp_path: Path) -> None:
    result = run_gang_command(_args(tmp_path, "show", "ghost"))
    assert not result.ok
    assert "not found" in result.message
