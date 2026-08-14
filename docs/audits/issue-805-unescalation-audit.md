# Issue #805: Audit of 7 merged PRs that were manually unescalated

## Method

For each of the 7 PRs (540, 531, 503, 584, 585, 637, 630), the audit followed the
method that worked for issue #627 / PR #700:

1. Read `.var/charlie-work/prs/pr-<N>/` — `review-decision.json`, `review-comment.md`,
   `review.md` / `review-summary.md` (where present), `cross-family-review.md`,
   `rework-prompt.md`, `reaped-verdict.json`, and `rework-dispatch-note.txt`.
2. Judged the review body, not the JSON. `required_changes: []` alongside a detailed
   review body is a known JSON-extraction gap (issue #792). Several of these PRs have
   `required_changes: []` in their final `review-decision.json` but substantive review
   bodies in sibling files.
3. Verified each substantive finding against the code at the reviewed SHA via
   `git show <sha>:<path>` — never `git checkout`.
4. Checked whether the defect still exists on `main` today.

All escalation events were queried from `events.db`:

```sql
SELECT * FROM events WHERE kind = 'unescalate' ORDER BY ts;
SELECT * FROM events WHERE kind LIKE '%escalat%' AND pr_number IN (540,531,503,584,585,637,630);
```

All 7 PRs share the same escalation reason: `max_review_dispatch_attempts_exceeded`
(the automated reviewer session failed 3 times — session limits, rate limits — not
because a substantive review found defects). The question for each is whether the
review body that led to or surrounded the escalation raised a substantive finding
that was never addressed before merge.

## Headline result

**0 of 7** shipped an unaddressed substantive finding.

All 7 PRs had prior review rounds with substantive findings (request_changes
verdicts), but every substantive finding was addressed via rework before merge.
Verified against code at the reviewed SHA and on current `main`.

| PR  | Issue | Escalation reason | Verdict | Substantive finding | Addressed? |
|-----|-------|-------------------|---------|---------------------|------------|
| 540 | #480  | max_review_dispatch_attempts_exceeded | SPURIOUS | `load_ledger` corruption guard gap | Yes — rework |
| 531 | #525  | max_review_dispatch_attempts_exceeded | SPURIOUS | `event_ring_size: 0` footgun | Yes — rework |
| 503 | #496  | max_review_dispatch_attempts_exceeded | SPURIOUS | `merge_hold` stripped by every transition | Yes — rework |
| 584 | #583  | max_review_dispatch_attempts_exceeded | SPURIOUS | (none — operator review) | N/A |
| 585 | #485  | max_review_dispatch_attempts_exceeded | SPURIOUS | (none — operator review, docs-only) | N/A |
| 637 | #633  | max_review_dispatch_attempts_exceeded | SPURIOUS | Third call site unguarded | Yes — rework |
| 630 | #623  | max_review_dispatch_attempts_exceeded | SPURIOUS | `require_global` absent-vs-invalid + merge conflict | Yes — rework |

No issues need to be filed — no substantive findings remain on `main`.

## Per-PR adjudication

### PR 540 (issue #480) — SPURIOUS

**Title:** feat(budget): api spend ledger with atomic settlement and budget status

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-24).
Unescalated 3 times (events 3953, 4310, 4808) on 2026-07-24 and 2026-07-25.

**Review body:** `reaped-verdict.json` shows `decision: request_changes` with one
substantive finding:

> `load_ledger`'s corruption guard only wraps `json.load()`; the subsequent
> `_ledger_from_dict(data)` call sits outside that try/except, so a
> syntactically-valid JSON file with a wrong-typed `usd` or `lifetime_usd` field
> raises uncaught, bypasses the documented quarantine path entirely, and silently
> wedges the ledger.

Two Minor non-blocking findings were also noted: (1) provider-pricing-miss silently
drops that session's cost with no log line, (2) `day_key==""` (malformed `started_at`)
branch is untested.

**Verification at reviewed SHA `dc59d2f`:** `git show dc59d2f:src/charlie_work/api_budget.py`
shows `load_ledger` now wraps `_ledger_from_dict(data)` inside the same
`try/except (json.JSONDecodeError, LookupError, ValueError, TypeError)` block — the
finding was addressed in the rework.

**Verification on current `main`:** `src/charlie_work/api_budget.py:452-479` — the
fix is present with an explanatory comment referencing the exact failure mode. The
two Minor findings remain (provider-pricing-miss has no log line; `day_key==""`
branch is still untested), but they were explicitly non-blocking and acknowledged in
the review.

**Verdict: SPURIOUS.** The escalation was mechanical (reviewer session failures).
The substantive finding was real but was addressed via rework before merge.

---

### PR 531 (issue #525) — SPURIOUS

**Title:** fix(state): raise event ring cap to 2000 and aggregate review dispatch events

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-24).
Unescalated once (event 3986) on 2026-07-24.

**Review body:** `review.md` shows `decision: request_changes` with one Important
finding:

> `event_ring_size: 0` is accepted by validation but produces unbounded ring growth.
> `load_config` validates `event_ring_size >= 0`, so `0` is a permitted configuration.
> But in `append_event`, `max_size=0` yields `events[-0:]`, and since `-0 == 0` in
> Python, `events[-0:]` is `events[0:]` — the entire list. An operator who sets
> `event_ring_size: 0` gets the opposite of what they intended: unbounded event
> growth — the exact failure mode this PR exists to prevent.

Three Minor findings: (1) negative-int validation branch untested, (2)
`OrchestratorApp.__init__` wiring has no regression test, (3) `numbers_key` selection
in `_append_sweep_events` is fragile for mixed payload shapes.

**Verification at reviewed SHA `b2f036b`:** `git show b2f036b:src/charlie_work/config.py`
shows validation now rejects `event_ring_size < 1` with an explicit comment about the
`-0 == 0` footgun — the finding was addressed in the rework.

**Verification on current `main`:** `src/charlie_work/config.py:2019-2039` — the fix
is present, with additional hardening (bool rejection at line 2025) added later.

**Verdict: SPURIOUS.** The escalation was mechanical. The substantive finding was
real but was addressed via rework before merge.

---

### PR 503 (issue #496) — SPURIOUS

**Title:** feat(auto-merge): merge-hold label to park approved PRs from mergequeue

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-24).
Unescalated once (event 4343) on 2026-07-24.

**Review body:** `review-summary.md` shows `decision: request_changes` with one
Important finding:

> `merge_hold` in `workflow_labels` is stripped by every transition, not just
> terminal ones. Every transition that uses `_compute_remove` — `review_started`,
> `rework_requested`, `review_approved`, `escalated`, `blocked`, `rework_dispatched`,
> `queued`, `dispatched`, `merged` — strips `merge_hold` from the issue unless
> `merge_hold` is in the add set, which it never is. This violates the issue's
> acceptance criterion: an operator who adds `merge_hold` to the linked issue will
> have it stripped on the next non-terminal transition, and the PR will be swept back
> into the mergequeue.

Three Minor findings: (1) `_record_merge_or_error` routing for
`merge_hold_check_unavailable` untested, (2) hold check scoped to mergequeue mode
only, (3) fail-closed on `issue_view` failure has broad blast radius.

**Verification at reviewed SHA `b427d75`:** `git show b427d75:src/charlie_work/config.py`
shows `merge_hold` is excluded from `workflow_labels` with an explicit comment — the
finding was addressed in the rework.

**Verification on current `main`:** `src/charlie_work/config.py:122-163` — the fix is
present with a detailed comment explaining the exclusion rationale.

**Verdict: SPURIOUS.** The escalation was mechanical. The substantive finding was
real but was addressed via rework before merge.

---

### PR 584 (issue #583) — SPURIOUS

**Title:** fix(review): count turn-limit deaths even when throttle tail is present

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-25).
Unescalated once (event 5384) on 2026-07-25.

**Review body:** `review-comment.md` shows "no verdict produced" — the automated
reviewer hit the session limit after 1 turn (0 tool calls). No rework-prompt.md
exists, meaning the PR was not sent back for rework.

The `review-decision.json` summary is an operator (human) review, not an automated
reviewer verdict:

> Operator review (reviewer sessions turn-limited 3x; escalation cap fired
> honestly). Fix verified against issue 583 live trace: turn-budget-exhausted deaths
> with throttle tails are now counted failures (claim -> review_dispatch_failed,
> attempt preserved, distinct `provider_throttled_turn_limit_counted` event) while
> the global quota backoff still applies; pure throttle deaths keep the rollback.

**Verification at reviewed SHA `9a601d8`:** `git show 9a601d8:src/charlie_work/workflow.py`
shows the `review_turn_limit_summary_posted` guard that counts turn-limit deaths as
failures even when a throttle tail is present.

**Verification on current `main`:** `src/charlie_work/workflow.py:3216` — the guard
is present. The `review_turn_limit_summary_posted` flag is set at line 11214 and
reset to `False` on every new claim at line 12271.

**Verdict: SPURIOUS.** The escalation was mechanical (reviewer session limit). The
operator manually reviewed and approved with a detailed verification against the
issue #583 live trace. No substantive finding was raised by the escalating review
(the automated reviewer produced no verdict).

---

### PR 585 (issue #485) — SPURIOUS

**Title:** docs(runbook): api worker operations, provider onboarding, calibration

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-25).
Unescalated once (event 6143) on 2026-07-26.

**Review body:** `review-comment.md` shows "no verdict produced" — the automated
reviewer hit the session limit after 1 turn (0 tool calls). No rework-prompt.md
exists.

The `review-decision.json` summary is an operator review:

> PR #585 adds a docs-only 'API worker operations' section to docs/RUNBOOK.md plus
> one CLAUDE.md invariant line, closing #485. Every concrete technical claim (config
> schema, routing preflight order/reasons, budget math including exactly-at-cap
> semantics, provider-auth-vs-throttle classification order, ledger
> settlement/quarantine, deep-merge scoping) was independently re-verified against
> the shipped source in config.py, routing.py, api_worker.py, api_budget.py,
> claude_code.py, state.py, global_config.py.

**Verification:** This is a docs-only PR (CLAUDE.md + docs/RUNBOOK.md). The operator
review verified every technical claim against the shipped source. No code changes to
verify against a reviewed SHA.

**Verdict: SPURIOUS.** The escalation was mechanical (reviewer session limit). The
operator manually reviewed and approved with a detailed verification of every
technical claim. No substantive finding was raised.

---

### PR 637 (issue #633) — SPURIOUS

**Title:** fix(github): merged_pr_list() must raise on unusable response, not silent-empty

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-26).
Unescalated once (event 8593) on 2026-07-28.

**Review body:** `rework-prompt.md` contains the orchestrator review with one
substantive finding:

> The core fix correctly root-causes and resolves the issue for the two guarded call
> sites (#502 tripwire and `_finalize_externally_merged_issues`), but the PR's claim
> that the third call site (`_resolve_merged_prs`'s direct fallback in
> `_dispatch_impl`) is unreachable in production is factually incorrect — it's the
> common-case path whenever there are open ready issues but no closed-ready issues
> this pass, and it has no error handling around the newly-raising
> `merged_pr_list()` call, with no test coverage.

**Verification at reviewed SHA `45581cb`:** `git show 45581cb:src/charlie_work/workflow.py`
shows `dispatch()` now has an `except GitHubError` handler that covers both the
direct-fallback fetch (branch 1) and the stored-error re-raise (branch 2), with
`_resolve_merged_prs` explicitly documented as propagating to that handler.

**Verification on current `main`:** `src/charlie_work/workflow.py:8133-8153` — the
`except GitHubError` handler is present with a detailed comment explaining both
branches. `_resolve_merged_prs` at line 8262-8278 propagates correctly.

**Verdict: SPURIOUS.** The escalation was mechanical. The substantive finding was
real but was addressed via rework before merge.

---

### PR 630 (issue #623) — SPURIOUS

**Title:** fix(config): make load_layered_config loud when the global layer is required

**Escalation:** `max_review_dispatch_attempts_exceeded` (3 attempts, 2026-07-28).
Unescalated once (event 9171) on 2026-07-28.

**Review body:** `rework-prompt.md` contains the orchestrator review with two
substantive findings:

1. Make the `require_global` fallback distinguish 'global absent/unreachable' (safe
   to retry without `require_global`) from 'global present but fails validation'
   (retrying hits the same `ConfigError` and still discards per-repo config) — or
   explicitly scope-note this as a known limitation and file a follow-up issue.
2. Add a regression test for a present-but-invalid global config under
   `require_global=True` proving the per-repo config either survives or the behavior
   is explicitly documented as out of scope.

Plus a merge conflict with `main`.

**Verification at reviewed SHA `b74e753`:** `git show b74e753:src/charlie_work/global_config.py`
shows `require_global` now raises a hard error when the global config is absent
(distinguishing absent from present-but-invalid), with a `ConfigError` rescue path
that retries with per-repo config alone when the merged load fails validation.

**Verification on current `main`:** `src/charlie_work/global_config.py:112-280` —
the fix is present. `require_global` raises on absent global (line 112), and the
`ConfigError` rescue path (line 258-281) retries with per-repo config alone when the
merged load fails, with a warning log and corrected provenance.

**Verdict: SPURIOUS.** The escalation was mechanical. The substantive findings were
real but were addressed via rework before merge.

---

## Comparison with the #627 / PR 700 case

Issue #627 / PR 700 was the one sampled case that came back positive: its escalation
fired mechanically (`request_changes_count(2) >= max_rework_cycles(2)`), but the
review behind it raised two real defects (supervisor lifecycle lock-release ordering
and unguarded `log_event` calls) that were never addressed before merge.

The key difference with these 7 PRs: their escalation reason was
`max_review_dispatch_attempts_exceeded` — the automated reviewer session failed to
produce a verdict at all (session limits, rate limits). For PRs 540, 531, 503, 637,
and 630, prior review rounds HAD produced substantive findings, but those findings
were addressed via rework (rework-prompt.md files exist, and the fixes are verified
at the reviewed SHAs and on current main). For PRs 584 and 585, the automated
reviewer never produced a verdict, and the operator manually reviewed and approved
with detailed verification.

In contrast, PR 700's escalation was `request_changes_count >= max_rework_cycles` —
the reviewer DID produce verdicts with substantive findings, but the rework cycle
cap was hit before they were fully addressed, and the manual unescalation bypassed
the remaining findings.

## Conclusion

The 627/700 case is an outlier, not a pattern. All 7 of the other manually-unescalated
merged PRs shipped without unaddressed substantive findings. The escalations were
mechanical (reviewer session failures), and where prior reviews had raised
substantive findings, those findings were addressed via rework before merge.

This does not diminish the forward fix proposed in the issue — making unescalation
visible to tooling via `correlation_id` and requiring a recorded reason. The audit
confirms the 627 case was the one that slipped through, and the structural fix
remains warranted to prevent the next one.
