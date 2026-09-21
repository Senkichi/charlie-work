8. Commit your changes with a Conventional-Commits message (`type(scope): description`).
9. Match CI locally before pushing: the ruff commands from step 7 plus, if the repository
   has a `.pre-commit-config.yaml`,
   `pre-commit run --files $(git diff --name-only origin/main...HEAD)`. Commit anything
   they fix — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
10. Push your branch: `git push -u origin $branch_name`.
11. Verify the push, draft your PR title/body (see PR requirements below), and write
    `.worker-outcome.json` per "Push and PR outcome" below -- do not run `gh pr create`;
    the orchestrator opens the PR from that file.
12. Finalize: confirm the working tree is clean (`git status --short`), the branch is
    pushed (`git branch -vv`), and `.worker-outcome.json` exists. Then stop -- do not
    wait for a PR to appear.
