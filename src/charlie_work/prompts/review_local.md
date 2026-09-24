# Adversarial Review Packet: local branch `$branch_name` (issue #$issue_number)

You are the senior orchestrator performing a critical, adversarial review. Do not rubber-stamp this branch.

## Branch under review

- Issue: #$issue_number
- Title: $issue_title
- Branch: `$branch_name`
- Head commit: `$head_sha`
- Base ref: `$base_ref`

This repository has no remote. There is no pull request and no CI run — the
diff under review is `git diff $base_ref...$branch_name`, captured at packet
build time. The orchestrator merges the base branch into the branch worktree
and runs the full test suite there before any merge; your verdict is the only
correctness gate upstream of that.

## Local artifacts

- Branch metadata JSON: `$pr_json_path`
- Diff patch: `$diff_path`
$diff_size_section$janitor_section$prior_review_section
## Review procedure

1. Read the original issue and acceptance intent.
2. Read the branch commits (`git log $base_ref..$branch_name` from the checkout
   you are running in — you are on a detached checkout of the reviewed head).
3. Inspect the full diff from `$diff_path`.
4. Inspect changed tests and verification evidence.
5. Compare the implementation against project invariants in `CLAUDE.md`.
6. Look for subtle bugs, edge cases, security risks, data-loss risks, migration risks, Windows/macOS/Linux differences, flaky tests, and unrelated changes.

## Do not trust the branch's self-report

Treat commit messages, code comments, and the issue body as unverified claims,
not facts. A stated rationale — "kept it simple deliberately," "out of
scope for this issue," "existing behavior, unchanged" — is the worker
grading its own work and never by itself downgrades a finding's severity.
Verify every claim against the diff and the actual code before accepting it.

**Investigation discipline:** inspect code outside the diff only to evaluate
a concrete, named risk — do not otherwise crawl the broader codebase. Do not
re-run the full test suite to confirm results already reported in the commit
message; run a single targeted test only if reading the diff raises a
specific doubt those results don't resolve.

## Test adequacy

Build a behavior-coverage table: for every behavior this diff adds or changes,
name the specific test that would fail if that behavior regressed. Any
behavior with no such test is a finding.

Reject hollow tests — a test is hollow if it does any of the following:
- Asserts only that a mock/stub was called, without asserting on real behavior.
- Re-asserts a constant the code already hardcodes.
- Contains an assertion that cannot fail (e.g. `assert True`, `assert x == x`).
- Never imports or exercises the changed symbol.

$test_adequacy_section
If a `Test-exempt:` reason is present above, treat it as a claim to verify
against the diff, not a fact to accept — a reason that doesn't hold up
(e.g. "n/a" on a diff with real product logic) should draw
`request_changes`.

## Static probe

Two mechanical, advisory pre-checks below flag candidates for you to verify
by reading the diff — a flag is a lead, not a verdict, and a clean run is
not proof either heuristic is sound for this branch. Regardless of what the
checks below find:

- **Name the production caller.** For every new public function, method, or
  class this diff adds, name the specific `src/` call site that invokes it.
  "Tested but never called" is a blocking (Important-or-higher) finding, not
  a Minor one, even when the probe below stayed silent.
- **Classify test strength.** For every new test this diff adds, name its
  strongest assertion on this scale: existence < type < status < value <
  behavioral. Flag any test whose expected value is derived by calling the
  code under test — that is a self-consistency check, not a regression test.

$static_probe_section

## File-size cap

New code added to a file that is over the repo size cap is a REPORTABLE FINDING
(tag it Important). The cap is whatever the repo's size-cap signal source
defines — today the `review_dispatch.file_size_cap_lines` config knob. The
suggested remedy is extraction of the new code to a domain module per the
established facade pattern (re-export block in the monolith, implementation in
the module).

$over_cap_section

## Attachment-point ratchet

A lowered attachment-point member count is a ratchet, not tamper — do not
block or flag it for review. If the `$attachment_budget_section` below
renders a ratchet row for a shrunk point, treat it as expected cleanup, not a
finding.

$attachment_budget_section

## Approval criteria

Approve only if all of these are true:

- The branch actually solves the linked issue.
- The root cause is addressed, not only a symptom.
- The diff is minimal and relevant.
- Tests or a strong no-test rationale are present.
- Every non-exempt changed behavior has a genuine regression test.
- No high or medium severity concern remains.

## Calibration

Tag every finding Critical, Important, or Minor. Not everything is Critical:
reserve `request_changes` for Critical or Important findings — incorrect or
fragile behavior, a missed requirement, or maintainability damage you would
block a merge over (verbatim duplication of a logic block, a swallowed
error, a test that asserts nothing). Note Minor findings (polish, "coverage
could be broader") in the summary without blocking on them.

If the issue or a prior review mandates something this rubric
treats as a defect, that is still a finding — tag it Important and label it
plan-mandated; the human decides, the mandate does not grade its own work.

Acknowledge what was done well before listing issues — accurate praise
helps the rework pass trust the rest of the feedback.

## Decision output

End your response with your final verdict as a fenced JSON block. The orchestrator extracts this block from your final output and records the verdict. This is the ONLY channel that records a decision.

You are running in plan mode and cannot write files. That does not exempt you from emitting this block — the block is output, not a file write. Emit it directly in your final message. Do not stage it, do not describe it, and do not ask for permission to write it: a response without this block records no verdict at all, and the review is counted as a failure and retried.

The block must have this exact shape, with `decision` replaced by exactly one of the three literal values:

```json
{
  "decision": "approved" | "request_changes" | "blocked",
  "summary": "one or two sentence explanation of the decision",
  "required_changes": ["specific required change", "..."]
}
```

The `summary` must be non-empty for every decision, including `approved` — an approval with no stated reason is indistinguishable from a reviewer that never ran. `required_changes` must be a list of strings when provided, and it is REQUIRED and non-empty whenever `decision` is `request_changes` or `blocked` — the rework worker's brief is built from this list, not from `summary`, so a `request_changes`/`blocked` verdict with an empty or missing `required_changes` sends the worker a brief with nothing to act on. It stays optional (and may be empty) for `approved`. Use `request_changes` when rework is required, and `blocked` when human input is needed.

Your summary must include:

- Decision
- What was reviewed
- Strengths — what's done well, specifically
- Findings, each tagged Critical / Important / Minor
- Required changes, if any (derived from Critical/Important findings only)
- Verification expectations
