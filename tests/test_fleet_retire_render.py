"""``charlie fleet retire`` / ``unretire`` command tests (issue #2754).

Split out of ``test_fleet_repo_retirement.py`` when the render-verification
tests pushed that module past the 800-line file-size cap (issue #1442).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work import cli
from charlie_work.command_result import CommandResult
from charlie_work.repo_retirement import repo_retirement_state


def test_fleet_retire_preserves_comments_and_unrelated_content(tmp_path: Path) -> None:
    """The command edits only the ``fleet.retired`` line, never a YAML dump."""
    import yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "# operator notes: do not reflow this file\n"
        "fleet:\n"
        "  retired: false  # flipped by hand during incidents\n"
        "  global_max_concurrent_sessions: 3\n"
        "\n"
        "# dispatch tuning\n"
        "dispatch:\n"
        "  default_limit: 2\n",
        encoding="utf-8",
    )
    before_text = config.read_text(encoding="utf-8")
    before = yaml.safe_load(before_text)

    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    result = cli.run_fleet_retire(args)

    assert result.ok is True
    after_text = config.read_text(encoding="utf-8")
    after = yaml.safe_load(after_text)
    assert after == {**before, "fleet": {**before["fleet"], "retired": True}}
    assert "# operator notes: do not reflow this file" in after_text
    assert "# dispatch tuning" in after_text
    assert "retired: true  # flipped by hand during incidents" in after_text
    changed = [
        (old, new)
        for old, new in zip(before_text.splitlines(), after_text.splitlines(), strict=True)
        if old != new
    ]
    assert changed == [
        (
            "  retired: false  # flipped by hand during incidents",
            "  retired: true  # flipped by hand during incidents",
        )
    ]


def test_fleet_retire_inserts_the_flag_into_an_existing_block(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "# keep me\nfleet:\n  global_max_concurrent_sessions: 3\n# below\n",
        encoding="utf-8",
    )

    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    assert cli.run_fleet_retire(args).ok is True

    lines = config.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "# keep me",
        "fleet:",
        "  global_max_concurrent_sessions: 3",
        "  retired: true",
        "# below",
    ]


def _retire_repo(repo: Path) -> CommandResult:
    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    return cli.run_fleet_retire(args)


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_fleet_retire_rewrites_a_single_line_flow_mapping_keeping_siblings(
    tmp_path: Path,
) -> None:
    """Issue #2754 rework: flow style must not drop the sibling keys."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet: {global_max_concurrent_sessions: 3, retired: false}\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    after = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert after == {
        "fleet": {"global_max_concurrent_sessions": 3, "retired": True},
        "dispatch": {"default_limit": 2},
    }
    assert repo_retirement_state(repo) is True


def test_fleet_retire_rewrites_a_multiline_flow_mapping(tmp_path: Path) -> None:
    """Issue #2754 rework: a multi-line flow mapping must stay parseable."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet: {\n"
        "  global_max_concurrent_sessions: 3,\n"
        "  retired: false,\n"
        "}\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    assert yaml.safe_load(config.read_text(encoding="utf-8")) == {
        "fleet": {"global_max_concurrent_sessions": 3, "retired": True},
        "dispatch": {"default_limit": 2},
    }
    assert repo_retirement_state(repo) is True


def test_fleet_retire_handles_a_column_zero_comment_before_the_key(tmp_path: Path) -> None:
    """A comment at column 0 inside the fleet block must not split it.

    Before the rework the edit inserted a second ``retired:`` above the
    comment; YAML accepted the duplicate and the last value won, so the
    command reported success while ``fleet.retired`` stayed false.
    """
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet:\n"
        "# operator note inside the fleet block\n"
        "  retired: false  # flipped during incidents\n"
        "  global_max_concurrent_sessions: 3\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    after_text = config.read_text(encoding="utf-8")
    assert repo_retirement_state(repo) is True
    assert "# operator note inside the fleet block" in after_text
    assert "retired: true  # flipped during incidents" in after_text
    assert after_text.count("retired:") == 1


def test_fleet_retire_appends_when_the_file_has_no_fleet_key(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text("dispatch:\n  default_limit: 2\n", encoding="utf-8")

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_text(encoding="utf-8") == (
        "dispatch:\n  default_limit: 2\n\nfleet:\n  retired: true\n"
    )
    assert repo_retirement_state(repo) is True


def test_fleet_retire_writes_an_empty_config_file(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text("", encoding="utf-8")

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_text(encoding="utf-8") == "fleet:\n  retired: true\n"
    assert repo_retirement_state(repo) is True


def test_fleet_retire_is_an_idempotent_no_op(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet:\n  retired: true\n  global_max_concurrent_sessions: 3\n"
    config.write_bytes(before)

    result = _retire_repo(repo)

    assert result.ok is True
    assert result.data["changed"] is False
    assert config.read_bytes() == before


def test_fleet_retire_preserves_crlf_newlines(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_bytes(
        b"fleet:\r\n  retired: false\r\n  global_max_concurrent_sessions: 3\r\n"
        b"dispatch:\r\n  default_limit: 2\r\n"
    )

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_bytes() == (
        b"fleet:\r\n  retired: true\r\n  global_max_concurrent_sessions: 3\r\n"
        b"dispatch:\r\n  default_limit: 2\r\n"
    )
    assert b"\r\r\n" not in config.read_bytes()


def test_fleet_retire_rejects_a_non_mapping_fleet_section(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet: [1, 2]\n"
    config.write_bytes(before)

    result = _retire_repo(repo)

    assert result.ok is False
    assert "fleet must be a mapping" in result.message
    assert config.read_bytes() == before


def test_fleet_retire_rejects_an_unreadable_config_file(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "orchestrator.config.yaml").mkdir()

    result = _retire_repo(repo)

    assert result.ok is False
    assert "unreadable config" in result.message


def test_fleet_retire_refuses_a_render_that_would_not_round_trip(tmp_path: Path) -> None:
    """The post-render verification is a write gate, not a decoration."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet:\n  retired: false\n  retired: false\n"
    config.write_bytes(before)
    # PyYAML accepts the duplicate and keeps the last value, so editing only
    # the first line would leave the effective value false.
    assert yaml.safe_load(before.decode())["fleet"]["retired"] is False

    result = _retire_repo(repo)

    assert result.ok is False
    assert "would not preserve" in result.message
    assert config.read_bytes() == before
