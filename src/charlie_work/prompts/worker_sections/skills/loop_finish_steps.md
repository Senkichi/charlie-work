8. Use `/commit` to commit your changes with conventional format.
9. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it
   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
10. Use `/push` to push your branch to GitHub.
11. Use `/create-pr` to create a pull request with proper formatting.
12. Use `/complete` to finalize the session.
