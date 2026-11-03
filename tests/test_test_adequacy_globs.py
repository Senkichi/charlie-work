"""Default ``test_path_globs`` classify nested and non-Python test files (issue #1707).

The defaults used to be ``("tests/**", "test_*.py", "*_test.py", "conftest.py")``,
matched with ``fnmatch`` against the full repo-relative path -- every glob
anchored at the repo root. A project nested under a subdirectory
(``server/tests/test_x.py``) or a non-pytest convention (Kotlin ``*Test.kt``,
bats, Pester) was therefore counted as *product* code, so the "product code
added but no test files changed" hard-fail fired on PRs that added tests.

These tests drive the real consumers (``janitor.check_test_adequacy`` and
``diff_coverage_probe.check_branch_coverage``) with the *config defaults*, not
hand-passed globs, so they fail if either default drifts back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _janitor_fixtures import _test_pr

from charlie_work.config import CoverageProbeConfig, TestAdequacyConfig, load_config
from charlie_work.diff_coverage_probe import check_branch_coverage
from charlie_work.janitor import check_test_adequacy
from charlie_work.test_adequacy_globs import DEFAULT_TEST_PATH_GLOBS

_PRODUCT_PATH = "server/app.py"


def _diff(path: str, added: list[str]) -> str:
    """A modify-diff for ``path`` that adds the given lines."""
    body = "".join(f"+{line}\n" for line in added)
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 123..456 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,1 +1,{len(added) + 1} @@\n"
        " existing\n"
        f"{body}"
    )


def _product_lines(count: int = 12) -> list[str]:
    return [f"value_{i} = {i}" for i in range(count)]


# Repo-relative paths that are test files by convention and must classify as test
# under the DEFAULTS. The first block is the original report (issue #1707 body);
# the rest are the layouts named in the issue comments.
_TEST_PATHS = [
    # -- nested Python (swole keeps everything under server/) ---------------
    pytest.param("server/tests/test_session_feedback_event.py", id="nested-pytest-new-file"),
    pytest.param("server/tests/test_wire_parity.py", id="nested-pytest-existing"),
    pytest.param("server/conftest.py", id="nested-conftest"),
    pytest.param("server/tests/conftest.py", id="nested-tests-conftest"),
    pytest.param("windows/tests/test_x.py", id="nested-windows-tests"),
    pytest.param("probes/tests/helpers.py", id="nested-tests-helper-not-test_-named"),
    pytest.param("server/pkg/foo_test.py", id="nested-suffix-test"),
    # -- root layouts keep working (no regression) --------------------------
    pytest.param("tests/test_x.py", id="root-tests-dir"),
    pytest.param("test_x.py", id="root-test-prefix"),
    pytest.param("conftest.py", id="root-conftest"),
    pytest.param("foo_test.py", id="root-suffix-test"),
    # -- Kotlin / KMP source sets under a project subdirectory --------------
    pytest.param(
        "swole/app/mobile/src/test/kotlin/dev/senki/swole/sync/SyncRepositoryTest.kt",
        id="kotlin-src-test",
    ),
    pytest.param(
        "swole/app/shared/src/commonTest/kotlin/dev/senki/swole/sync/BearerAuthInterceptorTest.kt",
        id="kotlin-commonTest",
    ),
    pytest.param(
        "swole/app/mobile/src/androidTest/kotlin/dev/senki/swole/FakeSyncApi.kt",
        id="kotlin-androidTest-helper-not-Test.kt-named",
    ),
    pytest.param("src/commonTest/kotlin/Fake.kt", id="kotlin-root-commonTest-helper"),
    pytest.param("app/FooTest.kts", id="kotlin-script-test"),
    # -- bats / Pester (Senkichi/hosts#90) ----------------------------------
    pytest.param("macbook/test/claude_config.bats", id="bats"),
    pytest.param("macbook/test/helpers.bash", id="bats-helper-in-test-dir"),
    pytest.param("env.bats", id="root-bats"),
    pytest.param("windows/steps/tests/Steps.Tests.ps1", id="pester-nested"),
    pytest.param("Root.Tests.ps1", id="pester-root"),
    # -- JS / TS ------------------------------------------------------------
    pytest.param("web/src/foo.test.ts", id="ts-test"),
    pytest.param("web/src/Foo.spec.tsx", id="tsx-spec"),
]

# Product paths that merely *look* test-adjacent. Misclassifying one as a test
# is the unsafe direction: it silently weakens the gate instead of failing
# visibly. Every entry is chosen so the verdict does not depend on filesystem
# case-folding (fnmatch case-folds on Windows).
_PRODUCT_PATHS = [
    pytest.param("server/app.py", id="plain-product"),
    pytest.param("server/contest/handler.py", id="contest-dir-is-not-test-dir"),
    pytest.param("server/latest_report.py", id="latest-is-not-test_-prefix"),
    pytest.param("server/testing_utils.py", id="testing_-is-not-test_-prefix"),
    pytest.param("api/openapi.spec.yaml", id="spec-yaml-is-product"),
    pytest.param("src/main/kotlin/dev/senki/Main.kt", id="kotlin-main-source-set"),
    pytest.param("web/src/app.js", id="plain-js"),
]


@pytest.mark.parametrize("test_path", _TEST_PATHS)
def test_default_globs_classify_test_path_as_test(test_path: str) -> None:
    """Product + a test file at ``test_path`` passes; the test file is NOT 'untested product'."""
    diff = _diff(_PRODUCT_PATH, _product_lines()) + _diff(test_path, ["assert True"])

    verdict = check_test_adequacy(diff, _test_pr(), TestAdequacyConfig())

    assert verdict.ok is True, verdict.failures
    assert verdict.facts.test_files_changed == 1
    assert tuple(verdict.facts.untested_product_files) == (_PRODUCT_PATH,)


@pytest.mark.parametrize("test_path", _TEST_PATHS)
def test_default_globs_dedicated_test_file_only_diff_is_not_product(test_path: str) -> None:
    """A diff that is *only* the test file adds zero product LOC (the PR #241 round-2 shape)."""
    diff = _diff(test_path, [f"    line_{i} = {i}" for i in range(12)])

    verdict = check_test_adequacy(diff, _test_pr(), TestAdequacyConfig())

    assert verdict.ok is True, verdict.failures
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 12
    assert tuple(verdict.facts.untested_product_files) == ()


@pytest.mark.parametrize("product_path", _PRODUCT_PATHS)
def test_default_globs_do_not_swallow_product_paths(product_path: str) -> None:
    """Control: a product-only diff on a look-alike path still hard-fails the gate."""
    diff = _diff(product_path, _product_lines())

    verdict = check_test_adequacy(diff, _test_pr(), TestAdequacyConfig())

    assert verdict.ok is False
    assert verdict.facts.test_files_changed == 0
    assert tuple(verdict.facts.untested_product_files) == (product_path,)


def test_explicit_override_replaces_defaults_not_merged() -> None:
    """No semantic change for operators: a configured tuple is used verbatim.

    The fix is in the *defaults* only. A repo that sets ``test_path_globs`` gets
    exactly its globs -- the new nested/multi-language defaults are not merged in,
    so existing overrides (e.g. swole's) keep their meaning.
    """
    diff = _diff(_PRODUCT_PATH, _product_lines()) + _diff(
        "server/tests/test_x.py", ["assert True"]
    )
    config = TestAdequacyConfig(test_path_globs=("custom_tests/**",))

    verdict = check_test_adequacy(diff, _test_pr(), config)

    assert verdict.ok is False
    assert verdict.facts.test_files_changed == 0


def test_defaults_applied_when_config_file_sets_no_override(tmp_path: Path) -> None:
    """swole shape: ``test_adequacy.enabled: true`` with no ``test_path_globs`` key."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("test_adequacy:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.test_adequacy.enabled is True
    assert config.test_adequacy.test_path_globs == DEFAULT_TEST_PATH_GLOBS
    assert config.coverage_probe.test_path_globs == DEFAULT_TEST_PATH_GLOBS


def test_both_gates_share_one_default_definition() -> None:
    """``CoverageProbeConfig`` used to carry its own copy of the defaults; it must not drift."""
    assert TestAdequacyConfig().test_path_globs == DEFAULT_TEST_PATH_GLOBS
    assert CoverageProbeConfig().test_path_globs == DEFAULT_TEST_PATH_GLOBS


def test_coverage_probe_defaults_treat_nested_tests_as_tests() -> None:
    """The advisory probe sees ``server/tests/test_*.py`` as test adds, not product branches."""
    product = _diff(_PRODUCT_PATH, ["def f(x):", "    if x:", "        return 1"])
    nested_test = _diff("server/tests/test_app.py", ["def test_f():", "    assert f(1) == 1"])
    config = CoverageProbeConfig()

    # Control: the product branch adds alone ARE flagged (the probe is live).
    alone = check_branch_coverage(product, config)
    assert [f.filename for f in alone] == [_PRODUCT_PATH]
    assert alone[0].reason == "no_test_adds"

    # With the nested test file in the diff the probe is silent.
    assert check_branch_coverage(product + nested_test, config) == ()
