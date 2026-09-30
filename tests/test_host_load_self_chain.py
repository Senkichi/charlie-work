"""Issue #2058: ``host_load._self_tree`` must not follow a stale parent link.

On Windows a recorded ``ParentProcessId`` is never updated when the parent
exits, and the freed pid can be recycled by an unrelated, *younger* process.
``_self_tree`` walks that ppid chain to find the topmost pytest root
containing the caller; trusting a recycled pid pulls an unrelated process --
and the pytest tree above it -- into the exclusion set, so a live suite stops
counting as external load.

(``tests/test_host_load.py``'s module attachment point is saturated, so the
#2058 cases live in this sibling file.)
"""

from __future__ import annotations

from charlie_work import host_load, quiesce


def _proc(
    pid: int, ppid: int, command_line: str, created: float | None = None
) -> quiesce.ProcessInfo:
    return quiesce.ProcessInfo(
        pid=pid, ppid=ppid, name="", command_line=command_line, created=created
    )


def test_pytest_tree_load_recycled_parent_does_not_exclude_strangers_suite() -> None:
    """300's recorded parent (200) exited and its pid was recycled by a
    process created *after* 300. Following the stale link would treat the
    stranger's pytest suite (100) as the tree containing self and exclude it
    -- reporting zero trees while a suite is running."""
    load = host_load.pytest_tree_load(
        (
            _proc(100, 1, "pytest -q", created=1_000.0),
            # The current holder of pid 200 is younger than its "child" 300.
            _proc(200, 100, "python -u -c xdist_worker", created=9_000.0),
            _proc(300, 200, "python -c caller", created=5_000.0),
        ),
        self_pid=300,
    )

    # The recycled link is rejected: 300 has no ancestor path into the suite,
    # so the suite counts as external load. 300 itself still counts as a
    # member -- its recorded ppid points into the tree, and the downward
    # subtree walk deliberately trusts recorded links (a stranger
    # self-attaching under a suite can only over-count, the conservative
    # direction for a load brake).
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=3)


def test_pytest_tree_load_consistent_chain_still_excludes_own_suite() -> None:
    """Control for the recycled-parent rule: with creation stamps present and
    every parent older than its child, the caller's own suite is still the
    excluded tree (topmost root) -- the fix narrows the exclusion, it does
    not remove it."""
    load = host_load.pytest_tree_load(
        (
            _proc(100, 1, "pytest -q", created=1_000.0),
            _proc(200, 100, "python -u -c xdist_worker", created=3_000.0),
            _proc(300, 200, "python -c test_body", created=5_000.0),
            _proc(400, 1, "pytest other/", created=2_000.0),  # unrelated -- counts
            _proc(401, 400, "python -u -c worker", created=2_100.0),
        ),
        self_pid=300,
    )

    # self's suite (100 -> 200 -> 300) is excluded; only the unrelated suite
    # remains.
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=2)


def test_pytest_tree_load_unknown_stamps_keep_the_chain() -> None:
    """Rows without a creation stamp keep their parent links -- unknown never
    disqualifies. POSIX snapshots never carry stamps (procfs reparents), so
    this is the whole-platform preserve-the-pre-fix-behavior case."""
    load = host_load.pytest_tree_load(
        (
            _proc(100, 1, "pytest -q"),
            _proc(200, 100, "python -u -c xdist_worker"),
            _proc(300, 200, "python -c test_body"),
            _proc(400, 1, "pytest other/"),
        ),
        self_pid=300,
    )

    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)
