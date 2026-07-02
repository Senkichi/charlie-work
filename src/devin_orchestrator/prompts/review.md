# Adversarial Review Packet: PR #$pr_number

You are the senior orchestrator performing a critical, adversarial review. Do not rubber-stamp this PR.

## PR

- Number: #$pr_number
- Title: $pr_title
- URL: $pr_url

## Linked issue

- Number: #$issue_number
- Title: $issue_title
- URL: $issue_url

## Local artifacts

- PR JSON: `$pr_json_path`
- Checks JSON: `$checks_json_path`
- Diff patch: `$diff_path`
$cross_family_section$janitor_section
## Review procedure

1. Read the original issue and acceptance intent.
2. Read the PR body and commits.
3. Inspect the full diff from `$diff_path`.
4. Inspect changed tests and verification evidence.
5. Inspect CI status from `$checks_json_path`.
6. Compare the implementation against project invariants in `CLAUDE.md`.
7. Look for subtle bugs, edge cases, security risks, data-loss risks, migration risks, Windows/macOS/Linux differences, flaky tests, and unrelated changes.

## Approval criteria

Approve only if all of these are true:

- The PR actually solves the linked issue.
- The root cause is addressed, not only a symptom.
- The diff is minimal and relevant.
- Tests or a strong no-test rationale are present.
- Required CI checks are passing or will be gated before merge.
- No high or medium severity concern remains.
- The PR body links the issue with `Closes #$issue_number` or equivalent.

## Decision output

Write your review summary to a Markdown file, then record one decision:

```powershell
$decision_command
```

Use `--decision request_changes` when rework is required. Use `--decision blocked` when human input is needed.

Your summary must include:

- Decision
- What was reviewed
- Findings
- Required changes, if any
- Verification expectations
