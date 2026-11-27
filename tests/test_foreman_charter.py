"""Charter validation: field checks, semantic checks, failure-as-value (issue #3117)."""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.foreman import paths
from charlie_work.foreman.charter import load_charter

REPOS = ("acme/widgets",)

MINIMAL = """\
name: {name}
outcome: Migrate the transport layer
repos: ["acme/widgets"]
"""

FULL = """\
name: {name}
outcome: Migrate the transport layer
repos: ["acme/widgets"]
standing: true
arming: propose
done_when:
  check: ["uv", "run", "pytest", "-q"]
bounds:
  max_open_issues: 7
  max_issues_per_turn: 2
  min_turn_interval_minutes: 15
  max_quiet_hours: 48
  max_turns_per_day: 6
  max_unanswered_days: 3
cadence:
  report_at: "09:30"
"""


def _foreman(tmp_path: Path, name: str = "g1") -> Path:
    fdir = paths.foreman_dir(tmp_path, name)
    fdir.mkdir(parents=True)
    return fdir


def _write(fdir: Path, text: str) -> None:
    (fdir / "charter.yaml").write_text(text, encoding="utf-8")


def test_minimal_charter_loads_with_defaults(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1"))
    result = load_charter(fdir, registered_repos=REPOS)
    assert result.ok, result.error
    charter = result.value
    assert charter.name == "g1"
    assert charter.repos == ("acme/widgets",)
    assert charter.standing is False
    assert charter.arming == "auto"
    assert charter.done_when is None
    assert charter.bounds.max_open_issues == 5
    assert charter.bounds.max_turns_per_day == 12
    assert charter.cadence.report_at is None


def test_full_charter_loads(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, FULL.format(name="g1"))
    charter = load_charter(fdir, registered_repos=REPOS).value
    assert charter.standing is True
    assert charter.arming == "propose"
    assert charter.done_when.check == ("uv", "run", "pytest", "-q")
    assert charter.bounds.max_quiet_hours == 48
    assert charter.cadence.report_at == "09:30"


def test_missing_charter_is_an_error_value(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "charter.yaml" in result.error


def test_invalid_yaml_is_an_error_value(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, "name: [unclosed\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "YAML" in result.error


def test_bad_name_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="Bad_Name"))
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "charter.name" in result.error


def test_name_must_match_directory(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="someone-else"))
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "must equal the foreman directory name" in result.error


def test_missing_outcome_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, 'name: g1\nrepos: ["acme/widgets"]\n')
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "charter.outcome" in result.error


def test_unregistered_repo_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1").replace("acme/widgets", "acme/other"))
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "not a registered repo" in result.error


def test_zero_and_two_repos_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, "name: g1\noutcome: x\nrepos: []\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "exactly one" in result.error
    _write(fdir, 'name: g1\noutcome: x\nrepos: ["acme/widgets", "acme/widgets"]\n')
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "exactly one" in result.error


def test_bad_arming_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + "arming: yolo\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "charter.arming" in result.error


def test_non_positive_bound_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + "bounds:\n  max_open_issues: 0\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "charter.bounds.max_open_issues" in result.error


def test_bad_report_at_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + 'cadence:\n  report_at: "25:61"\n')
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "report_at" in result.error


def test_done_when_without_check_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + "done_when:\n  check: []\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "done_when.check" in result.error


def test_unknown_key_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + "surprise: true\n")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "surprise" in result.error


def test_charter_symlink_escape_is_an_error_value(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text(MINIMAL.format(name="g1"), encoding="utf-8")
    try:
        (fdir / "charter.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert "escapes" in result.error


_BASE = 'name: g1\noutcome: Migrate the transport layer\nrepos: ["acme/widgets"]\n'


@pytest.mark.parametrize(
    "doc,expected",
    [
        # Every bounds field rejects a non-positive value.
        (_BASE + "bounds:\n  max_open_issues: 0\n", "charter.bounds.max_open_issues"),
        (_BASE + "bounds:\n  max_issues_per_turn: 0\n", "charter.bounds.max_issues_per_turn"),
        (
            _BASE + "bounds:\n  min_turn_interval_minutes: -1\n",
            "charter.bounds.min_turn_interval_minutes",
        ),
        (_BASE + "bounds:\n  max_quiet_hours: 0\n", "charter.bounds.max_quiet_hours"),
        (_BASE + "bounds:\n  max_turns_per_day: 0\n", "charter.bounds.max_turns_per_day"),
        (_BASE + "bounds:\n  max_unanswered_days: 0\n", "charter.bounds.max_unanswered_days"),
        # Field types: standing, repos, done_when.check, bounds, cadence.
        (_BASE + 'standing: "yes"\n', "charter.standing"),
        ("name: g1\noutcome: x\nrepos: acme/widgets\n", "charter.repos"),
        (_BASE + 'done_when:\n  check: "uv run pytest"\n', "charter.done_when.check"),
        (_BASE + "bounds: 5\n", "charter.bounds"),
        (_BASE + 'cadence: "09:00"\n', "charter.cadence"),
        # Empty outcome; missing or non-string name.
        ('name: g1\noutcome: ""\nrepos: ["acme/widgets"]\n', "charter.outcome"),
        ('outcome: x\nrepos: ["acme/widgets"]\n', "charter.name"),
        ('name: 5\noutcome: x\nrepos: ["acme/widgets"]\n', "charter.name"),
    ],
)
def test_charter_field_rejection_matrix(tmp_path: Path, doc: str, expected: str) -> None:
    """Every charter field's rejection path names the field — acceptance criterion."""
    fdir = _foreman(tmp_path)
    _write(fdir, doc)
    result = load_charter(fdir, registered_repos=REPOS)
    assert not result.ok
    assert expected in result.error


def test_done_when_null_is_preserved(tmp_path: Path) -> None:
    """``done_when: null`` and an omitted section land the same way (P2)."""
    fdir = _foreman(tmp_path)
    _write(fdir, MINIMAL.format(name="g1") + "done_when:\n")
    charter = load_charter(fdir, registered_repos=REPOS).value
    assert charter.done_when is None
