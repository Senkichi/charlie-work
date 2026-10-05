"""``datetime`` stand-in whose ``now`` reads the real clock plus an offset (HS-CW-5).

Patch it over a production module's ``datetime`` name around one call to
stamp "later" without sleeping across a whole-second boundary::

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "datetime", shifted_datetime(2))
        module.function_that_stamps()

Only ``now`` moves; constructors and every other classmethod are inherited.
A patched module that checks ``isinstance(x, datetime)`` would see real
datetimes fail that check -- read the target module before using this (the
HS-CW-5 targets, ``worktree`` and ``fleet_registry``, have no such check).
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo


def shifted_datetime(seconds: float) -> type[datetime]:
    offset = timedelta(seconds=seconds)

    class _Shifted(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return datetime.now(tz) + offset

    return _Shifted
