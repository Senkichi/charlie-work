8. Commit your changes with a Conventional-Commits message (`type(scope): description`).
9. Match CI locally before pushing: the ruff commands from step 7 plus, if the repository
   has a `.pre-commit-config.yaml`,
   `pre-commit run --files $(git diff --name-only origin/main...HEAD)`. Commit anything
   they fix — an uncommitted reflow or an un-normalized fixture is the #1 cause of a
   green-locally / red-on-CI PR, and the push/PR gate will block you on it.
10. Push your branch: `git push -u origin $branch_name`.
11. Open the pull request with `gh pr create` (see PR requirements below).
12. Finalize: confirm the working tree is clean (`git status --short`), the branch is
    pushed (`git branch -vv`), and the PR exists (`gh pr list --head $branch_name`).
