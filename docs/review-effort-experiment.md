# Reviewer-effort experiment

Issue #1701. The reviewer-effort A/B experiment has been assigning PRs to
arms since 2026-07-26; this document is the experiment's configuration
reference, its read-out, and its stopping rule.

## Configuration

The knobs live in the `reviewer` section of `orchestrator.config.yaml`
(`ReviewerRoleConfig` in `src/charlie_work/config.py`):

```yaml
reviewer:
  effort: high                     # effort for the treatment arm
  effort_experiment_fraction: 0.5  # share of PRs assigned to treatment
  effort_experiment_salt: ""       # re-randomization seed; see below
```

* `effort_experiment_fraction` in `(0, 1]` enables the experiment; `0.0`
  disables it. Validation requires `reviewer.effort` to be set when the
  fraction is nonzero — the treatment effort must exist to compare against.
* `_review_effort_arm(pr_number, fraction, salt)` in
  `src/charlie_work/claude_code.py` hashes `f"{salt}:{pr_number}"` with
  SHA-256, maps the first eight bytes to `[0, 1)`, and assigns the
  treatment arm below `fraction`. Assignment is deterministic and stable
  per PR across rework rounds and redispatches.
* With the experiment on, a treatment PR's reviewer runs with
  `reviewer.effort`; a control PR's reviewer runs with
  `claude_code.effort`. With the experiment off (`fraction: 0.0`), every
  PR uses `reviewer.effort` (falling back to `claude_code.effort` when
  `reviewer.effort` is unset).
* **Never change `effort_experiment_salt` mid-experiment** — it
  re-randomizes every assignment and silently mixes cohorts. Change it
  only to start a new experiment epoch.

## Unit of analysis: the PR, not the round

The arm value is a pure function of `(pr_number, salt)`, so every review
round of a PR lands in the same arm. Per-round rates therefore mislead:
each `request_changes` verdict spawns another round in the same arm, and a
round-level read-out would report kickback-heavy arms as worse without the
PR ever having been reviewed differently. `experiment-report` counts each
PR once for every rate.

## Reading the experiment

```bash
charlie experiment-report --experiment review_effort
```

`--experiment` selects the session-metrics key carrying the arm:
`review_effort` reads `review_effort_arm`; a value already ending in
`_arm` is used verbatim (the same command serves the planned `brief_arm`
experiment, issue #1276).

The command is **read-only**: it gates on `events.db` existing before
opening it (the instrumentation layer would otherwise create/migrate the
file), refuses when an unmigrated `events.jsonl` sits next to the
database, and writes nothing — `state.json` and `events.db` are
byte-identical before and after a run.

Window flags (all inclusive, ISO-8601; naive values read as UTC):

| flag | effect |
| --- | --- |
| `--since T` | only events at or after `T` contribute |
| `--until T` | only events at or before `T` contribute |
| `--exclude-window START END` | events inside `[START, END]` contribute to no figure; repeatable |
| `--json` | machine-readable report instead of text |
| `--min-prs-per-arm N` | analysis override for the stopping-rule minimum (the documented rule is fixed at 100) |

## Metric catalogue

Every metric carries a `measure` label:

* **activity** — what the arms *did*: first-round decision distribution,
  rounds per PR, review cost per PR, escalation share, merge share,
  dispatch-to-merge latency. Activity differences are descriptive only;
  they cannot decide the experiment.
* **outcome** — whether the review was *right*. These are the only metrics
  allowed to drive the stopping rule.

| metric | measure | definition |
| --- | --- | --- |
| `prs_assigned` | activity | PRs carrying a consistent arm value |
| `prs_with_rounds` | activity | assigned PRs with ≥1 recorded `record_review` |
| `first_round_{approved,request_changes,blocked}_rate` | activity | share of reviewed PRs whose **first** round had that decision |
| `rounds_per_pr_mean` | activity | mean recorded rounds per reviewed PR |
| `review_cost_per_pr_usd_mean` | activity | mean total `session_metrics.cost_usd` per reviewed PR |
| `escalation_rate` | activity | share of assigned PRs named by any `*_escalated` event |
| `merge_rate` | activity | share of assigned PRs reaching a merge event |
| `dispatch_to_merge_seconds_median` | activity | median dispatch→merge latency over merged PRs |
| `post_approval_defect_rate` | **outcome** | share of approved PRs later routed to rework by a post-approval defect signal (`check_failure_rework_requested`: required checks genuinely failed on the approved head; `cross_pr_revert_rework_requested`: the approved branch silently reverted a base commit) — an approval that did not hold up |
| `post_approval_ci_failure_rate` | outcome | the check-failure component alone |
| `post_approval_revert_bounce_rate` | outcome | the silent-revert component alone |
| `no_op_kickback_rate` | outcome | share of `request_changes` PRs whose next rework produced no change (`no_op_rework_repair_requested`) — a kickback that found nothing real to fix |

Each rate is reported with a 95% Wilson interval per arm and a pairwise
difference interval per arm pair; `n` is always shown so a wide interval
on a thin sample is visible, not hidden.

### Outcome coverage caveat

Two candidate outcomes from the issue are **not derivable** from recorded
events today, and the report says so explicitly rather than implying a
conclusion from activity metrics:

* post-merge main-branch CI failure attributed to the merged PR — no event
  links a post-merge main CI failure back to the PR;
* a follow-up fix or revert naming the merged PR — no event attributes a
  later fix/revert to a previously merged PR.

## Stopping rule

Evaluated by `experiment-report` against the scanned window; defaults are
`DEFAULT_MIN_PRS_PER_ARM = 100`, equivalence bound `= 2 × minimum`,
`CONFIDENCE_Z = 1.96` (`src/charlie_work/experiment_report.py`).

1. **Minimum N.** Every arm needs **≥ 100 assigned PRs** before anything
   is decided. Below that the rule reports `not met` — including a report
   with fewer than two arms.
2. **Difference detected.** Once the minimum holds, if the
   `post_approval_defect_rate` difference interval between any arm pair
   excludes 0, the experiment ends in favour of the lower-defect arm.
3. **No detectable difference.** Every arm at **≥ 200 assigned PRs** with
   all `post_approval_defect_rate` difference intervals still spanning 0
   ends the experiment as inconclusive-by-equivalence.
4. **Otherwise keep running** — the minimum is met but the outcome
   intervals still span 0 and the equivalence bound is not reached.

### What the configuration becomes afterward

When the rule reports `met` (either verdict):

* set `reviewer.effort_experiment_fraction: 0.0` — this turns the
  experiment off and makes `reviewer.effort` apply to every PR;
* for `difference_detected`: set `reviewer.effort` to the winning arm's
  effort — i.e. keep the current value if the treatment arm won, or set it
  equal to `claude_code.effort` (or unset it) if the control arm won;
* for `no_detectable_difference`: set `reviewer.effort` to whichever arm
  has the lower `review_cost_per_pr_usd_mean` — with no quality
  difference, cost decides.

### Inconclusive results

If the experiment is ended operationally before the rule is met (e.g. a
config or harness change makes the arms incomparable), treat it as
`no_detectable_difference`: set `effort_experiment_fraction: 0.0`, keep
the cheaper arm's effort, and record the cutoff window in the issue so a
future read-out can reproduce the cohort with
`--since`/`--until`/`--exclude-window`.
