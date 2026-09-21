"""Cross-repo gate regression fixtures from real swole issue bodies
(issues #1756-#1758).

Each fixture under ``tests/fixtures/cross_repo_gate_swole/`` is the verbatim
``body`` field of a real GitHub issue (fetched via ``gh issue view -R
Senkichi/swole <n> --json body``) that the pre-redesign "bare absence
escalates" rule would have false-positived: every one of these issues
references at least one file path that is absent from a fresh ``swole``
checkout (either a module-relative citation of a path nested deeper
elsewhere in the same repo, or a genuinely new file the issue proposes
adding), and none of those missing paths exist under any other managed
fleet repo either. Under the old rule, bare absence from ``repo_root`` was
itself treated as proof of a cross-repo target, so all five would have come
back ``passed=False`` (escalated) even though the subject code is either
already in ``swole`` (just not in this empty fixture checkout) or does not
exist anywhere yet. The positive-evidence redesign (see
``cross_repo_gate``'s module docstring and
``test_cross_repo_gate_sibling_repo.py``) requires a missing path to be
*positively found* under exactly one other managed repo's root before
escalating -- absence alone, from every repo, now abstains instead.

These tests exercise ``cross_repo_gate`` directly against a fake fleet
layout (a ``swole`` root plus a few empty sibling roots, mirroring
``test_cross_repo_gate_sibling_repo.py``'s style), not the dispatch-level
wiring -- that's covered by
``test_charlie_work_dispatch_blockers.py``/``test_charlie_work_dry_run.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.cross_repo_gate import cross_repo_gate

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cross_repo_gate_swole"


def _load_body(issue_number: int) -> str:
    path = FIXTURES_DIR / f"swole_issue_{issue_number}_body.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def fake_fleet(tmp_path: Path) -> dict[str, Path]:
    """A fake fleet: ``swole`` plus a few empty sibling repos.

    All roots are empty -- none of these five issues' missing paths are
    positive evidence of a cross-repo target under any of them, which is
    exactly the point: the old rule escalated on bare absence alone; the
    new rule needs a *positive* sibling hit, and there isn't one here.
    """
    roots = {}
    for name in ("swole", "charlie-work", "job-cannon", "fresh-eyes"):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    return roots


@pytest.mark.parametrize("issue_number", [92, 94, 213, 218, 276])
def test_real_swole_issue_body_does_not_false_positive(
    issue_number: int, fake_fleet: dict[str, Path]
) -> None:
    """Real swole issue #{issue_number}'s body must not escalate as
    cross-repo: every path it references is either genuinely new or a
    module-relative citation missing only from this empty fixture checkout,
    and none of them are positively found under any registered sibling.

    Would have failed before the redesign: the pre-#1756 gate treated bare
    absence from ``repo_root`` as sufficient evidence to escalate, so with
    an empty ``swole`` root every one of these issues' missing paths would
    have flipped ``passed`` to ``False`` regardless of the sibling repos'
    contents.
    """
    body = _load_body(issue_number)
    swole_root = fake_fleet["swole"]

    result = cross_repo_gate(body, swole_root, fake_fleet, "swole")

    assert result.passed is True
    assert result.found_in_repo is None
