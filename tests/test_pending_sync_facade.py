"""``supervise`` facade re-exports for the pending-sync domain (issue #1855).

The pending-sync marker helpers and starvation bound moved to
``charlie_work.pending_sync`` under the file-size ratchet (supervise.py is
over the 800-line module cap); ``supervise.py`` re-exports them so
pre-extraction imports and tests keep working. These checks pin that seam:
a re-declared copy in supervise.py would pass every behavioural test while
silently drifting from the module other consumers (``cli.py``,
``doctor_sync_starvation.py``) read -- the same failure shape
``test_ci_fleet_seams.py`` guards on the ci_fleet boundary.
"""

from __future__ import annotations

import pytest

import charlie_work.pending_sync as pending_sync
import charlie_work.supervise as supervise

_REEXPORTED_CALLABLES = [
    "_pending_sync_marker_path",
    "_write_marker",
    "_read_marker",
    "_parse_marker_timestamp",
    "_pending_sync_age_seconds",
    "_clear_marker",
    "record_sync_deferral",
]


@pytest.mark.parametrize("name", _REEXPORTED_CALLABLES)
def test_supervise_reexports_pending_sync_callables(name: str) -> None:
    """Each moved callable must *be* pending_sync's object, not a copy.

    A function re-declared in supervise.py is a second implementation:
    ``self_deploy`` would run one copy while callers importing from the
    domain module exercise another, and nothing fails loudly.
    """
    assert getattr(supervise, name) is getattr(pending_sync, name)


def test_supervise_reexports_the_result_type() -> None:
    """``SyncDeferral`` must be one class -- identity, not just equality."""
    assert supervise.SyncDeferral is pending_sync.SyncDeferral


def test_supervise_reexports_the_default_bound() -> None:
    """``cli.py`` reads ``DEFAULT_SYNC_STARVATION_SECONDS`` via supervise."""
    assert supervise.DEFAULT_SYNC_STARVATION_SECONDS == 14400
    assert (
        supervise.DEFAULT_SYNC_STARVATION_SECONDS is pending_sync.DEFAULT_SYNC_STARVATION_SECONDS
    )


def test_doctor_reads_the_emitted_kind() -> None:
    """The emit kind and the doctor query kind share one constant.

    ``record_sync_deferral`` emits ``self_deploy_sync_starved``;
    ``doctor_sync_starvation._check_sync_starvation`` queries it back. Two
    literals would drift silently -- the constant exists for the same
    reason label strings live in ``LabelConfig``.
    """
    import inspect

    import charlie_work.doctor_sync_starvation as doctor_mod

    assert pending_sync.SELF_DEPLOY_SYNC_STARVED_KIND == "self_deploy_sync_starved"
    source = inspect.getsource(doctor_mod)
    assert "SELF_DEPLOY_SYNC_STARVED_KIND" in source
    assert 'kind="self_deploy_sync_starved"' not in source
