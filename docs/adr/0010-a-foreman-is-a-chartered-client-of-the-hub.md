# A foreman is a chartered client of the hub, and may arm its own issues

Accepted 2026-11-25. The full design is `docs/design/2026-11-25-foreman-layer-design.md`.

charlie-work turns a Ready issue into a merged PR, and until now only a human turned an outcome into Ready issues. A **foreman** takes over that part for one outcome: it files the issues, arms them, watches them through the lifecycle, and raises items for the operator. It is built as a client of the orchestrator, in its own package (`charlie_work.foreman`), and it holds only the authority its **charter** grants.

A foreman's model call cannot write anything. It returns a list of **actions**, and deterministic code applies each one through an existing enforcement point or refuses it. This is the shape worker output and verdicts already have: LLM output enters the hub only as validated data.

Two things change that were true before:

- **Ready may be set by a foreman acting under its charter.** `CONTEXT.md` used to define Ready as something only a human marks. A foreman adds the Ready label to an issue it authored, as a separate final step, when its charter says `arming: auto`, it is not degraded, and the Turn read no outside signal. It never adds Ready to an issue someone else wrote.
- **The GitHub client can create an issue and edit an issue body.** Both backends gain the two requests. Only the foreman's Apply step calls them.

Everything else holds unchanged. The orchestrator has no LLM-driven control flow. Labels are the lifecycle source of truth (ADR-0001), and that now includes which issues a foreman owns: the ownership label is the truth and the foreman's local state is a cache. No module outside `charlie_work.foreman` imports it, and an import-graph test enforces that. The supervise loop knows one thing about the foreman: the argv that starts a foreman pass as a child process it does not wait on.

## Considered Options

- **A planning step inside the repo pass.** Rejected. It puts LLM control flow in the hub, makes a pass wait on a model, and ties fleet liveness to provider liveness.
- **A foreman with direct `gh` or shell access.** Rejected. Authority would live in a prompt, an injection through text on a public repo would have real reach, and none of the existing guards (issue-contract lint, outbound body guard, private-slug refusal) would sit on the path.
- **One long-lived interactive session per foreman.** Rejected. Context grows without bound, a dead session loses the plan, and the session holds the operator's whole tool surface. This is the way of working the foreman replaces.
- **A separate repository.** Rejected. The foreman needs the body parsers, the transport, the guards, the role chain and the event log at exactly the deployed commit. A subpackage with an enforced import direction gives the boundary without version skew.
- **A second scheduled task for the foreman pass.** Rejected. It would be registered by hand, could run a different version of the code than the hub during a self-deploy, and would keep running while the fleet is paused.

## Consequences

- A foreman's authority widens only when a human edits its charter. Model output cannot widen it.
- No action type exists for these effects, so a foreman can never perform them: recording a verdict, merging, unescalating, applying an escape hatch, editing a lifecycle label other than adding Ready to an issue it authored, amending or withdrawing an issue it did not author, changing config or its own charter, pushing code. When one of them is the right move, the foreman raises an item that carries the exact `charlie` command for the operator to run.
- The operator can hand an existing issue to a foreman by adding its ownership label. The foreman then tracks that issue and may comment on it, and can do nothing else to it.
- An operator edit to an issue a foreman authored blocks further amends by the foreman, which compares the live body against a hash of what it last wrote.
- A foreman is not an operator. Wherever the docs say "the operator arms issues", a foreman arming under its charter is the one delegated case; every other operator power stays with the human.
- The hub can be reasoned about, tested and deployed as if the foreman did not exist. Deleting the package and the one argv in the supervise loop removes the feature.
