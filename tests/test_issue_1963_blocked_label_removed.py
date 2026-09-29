"""Issue #1963: ``agent:blocked`` was a defined, bootstrapped, terminal label
that no transition ever applied -- the "blocked" verdict edge routes to
``agent:human-needed``, identical to "escalated". The dead ``LabelConfig``
field is removed; these tests pin the removal, the documented terminal set,
and the config-parse tolerance for a stale ``labels.blocked`` override.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from charlie_work.config import LabelConfig, load_config

REMOVED_LABEL = "agent:blocked"


def test_label_config_has_no_blocked_member() -> None:
    """No field, property member, or derived set may name ``agent:blocked`` --
    acceptance: no code path references ``labels.blocked``."""
    labels = LabelConfig()

    assert "blocked" not in {f.name for f in fields(LabelConfig)}
    assert not hasattr(labels, "blocked")
    assert REMOVED_LABEL not in labels.all
    assert REMOVED_LABEL not in labels.terminal
    assert REMOVED_LABEL not in labels.active
    assert REMOVED_LABEL not in labels.workflow_labels


def test_label_config_terminal_is_the_documented_set() -> None:
    """The terminal set equals the set documented in docs/RUNBOOK.md and
    docs/ARCHITECTURE.md -- no more, no less."""
    labels = LabelConfig()

    assert labels.terminal == {
        labels.done,
        labels.human_needed,
        labels.prose_only_deps,
        labels.operator_queue,
        labels.review_ready,
    }


def test_no_label_edge_adds_the_removed_label() -> None:
    """Every ``labels._edges`` add-list must be free of ``agent:blocked`` --
    the label was never applied, and nothing may resurrect it."""
    from charlie_work.labels import _edges

    labels = LabelConfig()
    for event, (add, _remove) in _edges(labels).items():
        assert REMOVED_LABEL not in add, f"edge {event!r} re-applies {REMOVED_LABEL}"


def test_blocked_edge_still_routes_to_human_needed() -> None:
    """Only the dead *label* was removed -- the ``blocked`` *verdict* edge is
    live and still lands on ``agent:human-needed``."""
    from charlie_work.labels import _edges

    labels = LabelConfig()
    add, _remove = _edges(labels)["blocked"]

    assert add == (labels.human_needed,)


def test_labels_section_tolerates_stale_blocked_key(tmp_path: Path) -> None:
    """A live config that still carries ``labels: {blocked: ...}`` must not
    trip ``_build_section``'s unknown-key ``ConfigError`` -- that would brick
    the repo on self-deploy. The dead override is silently ignored while
    sibling keys still parse."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """labels:
  ready: custom-ready
  blocked: agent:blocked
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.labels.ready == "custom-ready"
    assert not hasattr(config.labels, "blocked")
