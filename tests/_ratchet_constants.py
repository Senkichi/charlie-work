"""Shared test-side constant for the file-size ratchet surface.

``MARK_QUANTUM`` is authoritatively declared in
``scripts/refresh_file_size_ratchet.py`` (the SOLE writer of
``file_size_ratchet_baseline.json``). The test side needs the same value --
``tests/test_file_size_ratchet.py`` uses it in the keystone's remedy text and
mutation checks -- but test modules may not import each other
(``tests/test_zero_cross_test_import_guard.py``), and the script is only
reachable via ``tests/_script_loader``. So the test-side copy lives here, in
an underscore-prefixed shared module, and
``tests/test_refresh_file_size_ratchet.py`` asserts the script's constant
equals this one, so the two cannot silently drift.
"""

MARK_QUANTUM = 200
