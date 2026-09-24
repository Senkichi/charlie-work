# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for every operation;
`gh` infers the repo from the clone.

## Conventions

- Create: `gh issue create --title "..." --body-file <file>`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open --json number,title,body,labels`
- Comment: `gh issue comment <number> --body "..."`
- Labels: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

## Pull requests as a triage surface

No. Pull requests here are opened by the fleet orchestrator for its own workers; `/triage`
ignores them.

## Blockers (fleet contract; overrides any skill default)

The fleet dispatcher reads blockers from the issue body. GitHub's native issue dependencies are only a best-effort extra. For every issue that must
wait for another:

- Put a `## Blocked by` heading in the body and, under it, one line per blocker of the exact
  form `Blocked by #<n>`, with nothing else on the line.
- Adding the native dependency as well is fine; it never replaces the line.
- Never express an ordering only in prose (for example "after the parser lands"): the fleet
  cannot read it and parks the issue for a human.
- An issue with no blockers writes `None` under the heading.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
