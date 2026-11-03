"""``selection_answer``: the once-per-pass memo and the durable change baseline (issue #2921).

``worker_selection`` runs for every candidate on every pass. The memo makes the repo's
answer a once-per-pass fact; the baseline makes the fallback event a once-per-change
fact. The behaviour through ``worker_selection`` itself is pinned in
``tests/test_selection_wrapper_probe.py``; these tests pin the two halves in isolation,
including the failure directions the module docstring promises (a bookkeeping failure
never swallows a report and never raises into a prompt render).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from charlie_work import layout, selection_answer
from charlie_work.instrumentation import correlation_context
from charlie_work.state import StateLockBusy

FULL = ("selection-full", "ci-fleet select answers full (reason: map-stale)")
NO_MAP = ("selection-full", "ci-fleet select answers full (reason: no-map)")
NO_EXE = ("no-executable", "no ci-fleet.exe beside python")


@pytest.fixture(autouse=True)
def _empty_memo() -> None:
    """Tests must not share the module-level memo."""
    selection_answer._memo.clear()


@pytest.fixture
def baseline(tmp_path: Path) -> Path:
    return layout.worker_selection_answer_path(tmp_path)


class Counter:
    """A ``compute`` that returns a distinct value per call, so reuse is visible."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.calls


# ---------------------------------------------------------------------------
# 1. once_per_pass
# ---------------------------------------------------------------------------


def test_the_value_is_computed_once_per_pass_and_key() -> None:
    compute = Counter()
    with correlation_context():
        first = selection_answer.once_per_pass("repo-a", compute)
        again = selection_answer.once_per_pass("repo-a", compute)
        other = selection_answer.once_per_pass("repo-b", compute)
    assert (first, again, other) == (1, 1, 2)
    assert compute.calls == 2


def test_a_new_pass_computes_again() -> None:
    compute = Counter()
    with correlation_context():
        selection_answer.once_per_pass("repo-a", compute)
    with correlation_context():
        assert selection_answer.once_per_pass("repo-a", compute) == 2


def test_outside_a_pass_nothing_is_memoized() -> None:
    """No correlation id means no pass to scope a memo to: every call resolves."""
    compute = Counter()
    assert [selection_answer.once_per_pass("repo-a", compute) for _ in range(3)] == [1, 2, 3]
    assert selection_answer._memo == {}


def test_a_falsy_value_is_memoized_too() -> None:
    """``worker_selection`` returns None for a fallback; None must not read as a miss."""
    calls: list[None] = []

    def compute() -> None:
        calls.append(None)

    with correlation_context():
        selection_answer.once_per_pass("repo-a", compute)
        selection_answer.once_per_pass("repo-a", compute)
    assert len(calls) == 1


def test_a_failing_compute_memoizes_nothing() -> None:
    attempts: list[int] = []

    def compute() -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return 42

    with correlation_context():
        with pytest.raises(RuntimeError):
            selection_answer.once_per_pass("repo-a", compute)
        assert selection_answer.once_per_pass("repo-a", compute) == 42


def test_the_memo_is_bounded_and_evicts_the_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selection_answer, "MEMO_LIMIT", 3)
    compute = Counter()
    with correlation_context():
        for key in ("a", "b", "c", "d"):
            selection_answer.once_per_pass(key, compute)
        assert len(selection_answer._memo) == 3
        # "a" was evicted, so it computes again; "d" is still held.
        assert selection_answer.once_per_pass("a", compute) == 5
        assert selection_answer.once_per_pass("d", compute) == 4


# ---------------------------------------------------------------------------
# 2. settle_answer: the change table
# ---------------------------------------------------------------------------


def test_a_usable_answer_with_nothing_recorded_writes_nothing(baseline: Path) -> None:
    assert selection_answer.settle_answer(baseline, None) is False
    assert not baseline.exists()


def test_the_first_unavailable_answer_reports_and_a_repeat_does_not(baseline: Path) -> None:
    assert selection_answer.settle_answer(baseline, FULL) is True
    assert selection_answer.load_answer(baseline) == FULL
    first_write = baseline.read_text(encoding="utf-8")
    assert selection_answer.settle_answer(baseline, FULL) is False
    assert selection_answer.settle_answer(baseline, FULL) is False
    assert baseline.read_text(encoding="utf-8") == first_write, "a repeat must not rewrite"


def test_a_changed_detail_or_reason_reports_once_each(baseline: Path) -> None:
    assert [
        selection_answer.settle_answer(baseline, answer)
        for answer in (FULL, NO_MAP, NO_MAP, NO_EXE, NO_EXE, FULL)
    ] == [True, True, False, True, False, True]


def test_a_usable_answer_clears_the_baseline_so_the_next_fallback_reports(baseline: Path) -> None:
    assert selection_answer.settle_answer(baseline, FULL) is True
    assert selection_answer.settle_answer(baseline, None) is False
    assert selection_answer.load_answer(baseline) is None
    assert selection_answer.settle_answer(baseline, None) is False
    assert selection_answer.settle_answer(baseline, FULL) is True


def test_the_document_is_versioned_and_timestamped(baseline: Path) -> None:
    selection_answer.settle_answer(baseline, FULL)
    document = json.loads(baseline.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert document["answer"] == {"reason": FULL[0], "detail": FULL[1]}
    assert document["generated_at"].endswith("Z")


# ---------------------------------------------------------------------------
# 3. Fail-safe direction
# ---------------------------------------------------------------------------

_UNUSABLE_FILES = [
    pytest.param("{not json", id="malformed-json"),
    pytest.param("[]", id="not-an-object"),
    pytest.param(json.dumps({"answer": "selection-full"}), id="answer-not-an-object"),
    pytest.param(json.dumps({"answer": {"reason": "r"}}), id="answer-missing-detail"),
    pytest.param(json.dumps({"answer": {"reason": 1, "detail": "d"}}), id="reason-not-a-string"),
    pytest.param(json.dumps({"version": 1}), id="no-answer"),
    pytest.param("", id="empty-file"),
]


@pytest.mark.parametrize("text", _UNUSABLE_FILES)
def test_an_unusable_baseline_reads_as_nothing_reported_and_heals(
    baseline: Path, text: str
) -> None:
    baseline.write_text(text, encoding="utf-8")
    assert selection_answer.load_answer(baseline) is None
    assert selection_answer.settle_answer(baseline, FULL) is True
    assert selection_answer.load_answer(baseline) == FULL
    assert selection_answer.settle_answer(baseline, FULL) is False


def test_a_utf8_bom_is_tolerated(baseline: Path) -> None:
    baseline.write_text(
        json.dumps({"answer": {"reason": FULL[0], "detail": FULL[1]}}),
        encoding="utf-8-sig",
    )
    assert selection_answer.load_answer(baseline) == FULL
    assert selection_answer.settle_answer(baseline, FULL) is False


@pytest.mark.parametrize("failure", [StateLockBusy("busy"), OSError("read-only state dir")])
@pytest.mark.parametrize(("answer", "reports"), [(FULL, True), (None, False)])
def test_a_failed_write_still_reports_an_unavailable_answer(
    monkeypatch: pytest.MonkeyPatch,
    baseline: Path,
    failure: BaseException,
    answer: selection_answer.Answer | None,
    reports: bool,
) -> None:
    """Telemetry is not lost to a bookkeeping failure, and a usable answer never reports."""
    # A prior ``FULL`` makes ``None`` a real change, so the write path is reached for both.
    selection_answer.settle_answer(baseline, FULL if answer is None else NO_MAP)

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("charlie_work.state.advisory_file_lock", fail)
    assert selection_answer.settle_answer(baseline, answer) is reports


def test_a_permission_error_on_the_read_never_raises_and_reads_as_nothing_reported(
    monkeypatch: pytest.MonkeyPatch, baseline: Path
) -> None:
    real_open = Path.open

    def deny(self: Path, *args: object, **kwargs: object):
        if self == baseline:
            raise PermissionError(13, "denied", str(self))
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", deny)
    assert selection_answer.load_answer(baseline) is None
    assert selection_answer.settle_answer(baseline, FULL) is True
    assert selection_answer.settle_answer(baseline, None) is False


# ---------------------------------------------------------------------------
# 4. Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_observers_of_one_new_answer_report_it_exactly_once(baseline: Path) -> None:
    """The comparison is repeated under the lock, so racing settlers cannot both report."""
    workers = 8
    barrier = threading.Barrier(workers)

    def settle() -> bool:
        barrier.wait()
        return selection_answer.settle_answer(baseline, FULL)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: settle(), range(workers)))
    assert results.count(True) == 1
    assert selection_answer.load_answer(baseline) == FULL


# ---------------------------------------------------------------------------
# 5. layout
# ---------------------------------------------------------------------------


def test_layout_places_the_baseline_beside_state_json(tmp_path: Path) -> None:
    assert layout.worker_selection_answer_path(tmp_path) == (
        tmp_path / layout.WORKER_SELECTION_ANSWER_FILENAME
    )
    assert layout.WORKER_SELECTION_ANSWER_FILENAME == "worker-selection-answer.json"
