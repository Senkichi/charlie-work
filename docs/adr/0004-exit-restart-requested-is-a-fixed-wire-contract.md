# EXIT_RESTART_REQUESTED is a fixed wire contract

`supervise_loop.EXIT_RESTART_REQUESTED` (3) is the exit code by which a `fleet supervise` child tells its `fleet supervise-loop` wrapper to replace it with the new code on disk, and its value must never change. The wrapper imports the constant at startup and keeps it in memory, while the child it spawns loads it fresh from disk. A self-deploy is exactly when those two run different commits, so renumbering would make a stale wrapper read a restart request as a normal exit and skip the relaunch.

The wrapper was added by PR #888 for issue #862. Before it, a self-deploy left the fleet with no supervisor for up to a full 5-minute watchdog interval, because only the scheduled-task tick relaunched it.

## Considered Options

A runtime handshake between wrapper and child was not built. The comment in `supervise_loop.py` explains that recovery from a mismatch is bounded: the wrapper exits and the next 5-minute tick starts a fresh one. That makes a documented invariant enough, on the same discipline `CLAUDE.md` applies to label strings.

## Consequences

- The constant is defined once. The launcher script never compares against the literal, and the tests and `cli.py` import the symbol.
- `test_exit_restart_requested_literal_is_pinned` in `tests/test_supervise_loop.py` pins the literal `3`. Every other test uses the constant by name, so without this one a renumbering would pass them all.
- Other exit codes must not collide with it. `PREFLIGHT_REFUSAL_EXIT_CODE` (4, #1363) is a separate value because reusing 3 would make a fresh wrapper relaunch a process that had just refused to start.
- `python -m charlie_work.cli` must return `main()`'s code through `SystemExit(main())`. A bare `main()` call exited 0, so a restart request would read as a clean exit, which is the #862 outage again (#959, `cli.py`).
- The wrapper's relaunch count is capped (`DEFAULT_MAX_RELAUNCHES = 10`). Hitting the cap returns restart authority to the scheduled tick.
