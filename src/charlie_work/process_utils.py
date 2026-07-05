"""Shared utilities for process management across adapters."""

from __future__ import annotations


def parse_proc_stat_starttime(stat_text: str) -> int | None:
    """Parse /proc/<pid>/stat to extract the starttime field.

    Args:
        stat_text: The content of /proc/<pid>/stat as a string.

    Returns:
        The starttime value in clock ticks since boot, or None if parsing fails.

    The /proc/<pid>/stat format is:
        pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
        cmajflt utime stime cutime cstime priority nice num_threads itrealvalue
        starttime vsize rss rsslim startcode endcode startstack kstkesp kstkeip
        signal blocked sigignore sigcatch wchan ...

    The comm field can contain spaces and parentheses, so we must split on the
    LAST ')' to correctly separate it from the remaining fields. Using rpartition
    ensures we handle embedded ')' characters in comm (e.g., "(tmux: (0) server)").
    """
    # Split on the LAST ')' to handle comm containing embedded ')'
    before_paren, sep, after_paren = stat_text.rpartition(")")
    if not sep:
        # No ')' found - malformed stat line
        return None

    # After the comm field, we have the remaining space-separated fields
    after_comm = after_paren.strip().split()

    # starttime is at index 19 (0-indexed) after comm
    # Format: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
    #         cmajflt utime stime cutime cstime priority nice num_threads
    #         itrealvalue starttime ...
    if len(after_comm) < 20:
        return None

    try:
        return int(after_comm[19])
    except (ValueError, IndexError):
        return None
