"""Topmost-root folding tests for ``pytest_tree_load`` (issue #1918).

Sibling of ``tests/test_host_load.py`` -- that module's attachment point
is saturated, so the #1918 regression tests live here (the same split
pattern as ``test_ci_headroom.py`` / ``test_ci_capacity_backpressure.py``
and this module's other sibling ``test_host_load_tree_headroom.py``).
The parent module pins the measurement layer's happy path; this one pins
the defect: ``pytest_tree_load`` iterated ``root_pids`` in hash order,
which for ints is roughly ascending PID, so a nested pytest root with a
*lower* PID than its outer root was visited first and counted as its own
tree -- and the outer root, not yet in ``seen``, counted again. Real
Windows PIDs are effectively random, so live wrapper chains
(``bash -c`` -> ``timeout`` -> ``uv run`` -> ``python -m pytest``)
overcounted 3-4 matching roots per real suite and deferred dispatch on
phantom ``host_load_max_pytest_trees`` load.

A tree now counts once per *topmost* root -- a root with no matching
root anywhere on its ancestor chain, resolved through the same
cycle/PID-reuse-guarded ``ppid`` walk ``_self_tree`` uses. Nested roots
fold into their topmost ancestor's tree regardless of PID ordering.
"""

from __future__ import annotations

import random

from charlie_work import host_load, quiesce

# Self-contained per the zero-cross-test-import guard
# (tests/test_zero_cross_test_import_guard.py): these helpers duplicate
# test_host_load.py's _proc/_procs rather than importing them.


def _proc(pid: int, ppid: int, command_line: str, name: str = "") -> quiesce.ProcessInfo:
    return quiesce.ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line)


def _procs(*procs: quiesce.ProcessInfo) -> tuple[quiesce.ProcessInfo, ...]:
    return tuple(procs)


def test_nested_root_with_lower_pid_folds_into_outer_tree() -> None:
    """Regression for #1918: the nested fixture from ``test_host_load.py``
    with the PIDs inverted -- outer root 500, nested root 100 under a
    non-root intermediate. Hash order visited 100 first and counted it as
    its own tree (then 500 again); a nested root must fold into its
    topmost ancestor's tree, yielding the same counts as the
    ascending-PID case."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(500, 1, "pytest -q"),
            _proc(300, 500, "python -u -c worker"),  # non-root intermediate
            _proc(100, 300, "python -m pytest inner/"),  # nested root, lower pid
            _proc(50, 100, "python -u -c inner_worker"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=4)


def test_wrapper_chain_in_descending_pid_order_counts_once() -> None:
    """The live shape from the issue: a four-deep wrapper chain where
    EVERY layer's command line matches the invocation regex
    (``bash -c`` -> ``timeout`` -> ``uv run`` -> ``python -m pytest``),
    in descending-PID order -- the ordering that made hash-order
    iteration count each layer as its own tree. These PIDs put the
    topmost root LAST under int-set iteration, so unfixed code counts
    all four layers as separate trees."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(9999, 1, 'bash -c "cd repo && uv run --extra dev pytest -q"'),
            _proc(8888, 9999, "timeout 1500 uv run --extra dev pytest -q"),
            _proc(7777, 8888, "uv run --extra dev pytest -q"),
            _proc(1111, 7777, "python -m pytest -q --tb=short"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=4)


def test_tree_count_is_snapshot_order_independent() -> None:
    """The reduction must not depend on the snapshot's ordering: every
    permutation of the same processes yields the same ``HostLoad`` --
    topmost-root resolution is a pure function of the ``ppid`` graph,
    not of iteration order."""
    procs = [
        _proc(9999, 1, 'bash -c "cd repo && uv run --extra dev pytest -q"'),
        _proc(8888, 9999, "timeout 1500 uv run --extra dev pytest -q"),
        _proc(7777, 8888, "uv run --extra dev pytest -q"),
        _proc(1111, 7777, "python -m pytest -q --tb=short"),
        _proc(50, 1, "pytest -n 2"),  # a second, disjoint suite
        _proc(51, 50, "python -u -c xdist"),
        _proc(60, 50, "python -u -c xdist"),
    ]
    expected = host_load.HostLoad(pytest_tree_count=2, pytest_process_count=7)
    assert host_load.pytest_tree_load(procs, self_pid=999) == expected
    rng = random.Random(0)
    for _ in range(25):
        shuffled = procs[:]
        rng.shuffle(shuffled)
        assert host_load.pytest_tree_load(shuffled, self_pid=999) == expected


def test_self_exclusion_follows_topmost_root_with_inverted_pids() -> None:
    """Self-exclusion with inverted PIDs: self sits under a nested root
    whose pid is lower than its outer root's. The whole OUTERMOST tree
    is still excluded -- the same shape as
    ``test_pytest_tree_load_excludes_topmost_ancestor_tree`` with the
    PID ordering that used to double-count."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(500, 1, "pytest outer"),
            _proc(100, 500, "python -m pytest inner"),  # nested root, lower pid
            _proc(90, 100, "python test_body"),  # self_pid
            _proc(95, 500, "python -u -c outer_worker"),  # sibling of inner
            _proc(600, 1, "pytest other_suite"),  # unrelated -- counts
        ),
        self_pid=90,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)
