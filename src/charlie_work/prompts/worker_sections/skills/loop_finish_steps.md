8. Use `/commit` to commit your changes with conventional format.
9. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it
   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
10. Use `/push` to push your branch to GitHub.
11. Verify the push, draft your PR title/body (see PR requirements below), and write
    `.worker-outcome.json` per "Push and PR outcome" below -- do not create a pull
    request yourself; the orchestrator opens it from that file.
12. Use `/complete` to finalize the session.
