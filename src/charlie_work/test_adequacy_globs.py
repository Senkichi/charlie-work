"""Default ``test_path_globs`` shared by the test-adequacy gate and the coverage probe.

Issue #1707. Classification of a diff file as "test" is
``fnmatch.fnmatch(<full repo-relative path>, glob)`` (``janitor``,
``diff_coverage_probe``, ``worktree.summarize_branch_work``). ``fnmatch`` has
no path semantics: ``*`` and ``**`` both match any characters *including* ``/``,
and a glob is anchored at both ends of the whole path. Two consequences drive
the shape of the defaults below:

* A glob that begins with a literal (``tests/**``, ``test_*.py``,
  ``conftest.py``) only matches at the repo root. A project nested under a
  subdirectory (``server/tests/test_x.py``) is invisible to it, so every such
  file was counted as product code and the "product code added but no test
  files changed" hard-fail fired on essentially every PR that added tests.
* ``**/x`` does not match a root-level ``x`` -- it needs a literal ``/`` before
  ``x``. So each directory/prefix convention is listed twice: once rooted, once
  as ``**/...``. A trailing-wildcard suffix glob (``*_test.py``, ``*Test.kt``)
  already crosses ``/`` and needs no twin.

The set is deliberately conservative in the *safe* direction. Over-matching
quietly turns product code into "test" and weakens the gate; under-matching
fails visibly and has the ``Test-exempt:`` escape hatch plus a per-repo
``test_adequacy.test_path_globs`` override. So patterns are anchored on a
directory name or a language-specific extension rather than a bare substring:
``**/test_*.py`` (not ``*test_*.py``, which would match ``latest_x.py``) and
``*.spec.[jt]s`` (not ``*.spec.*``, which would match ``openapi.spec.yaml``).

Operators override the whole tuple per repo; this is only the default. Note
that ``fnmatch.fnmatch`` case-folds on Windows (``os.path.normcase``), so a
case-sensitive glob such as ``*Test.kt`` also matches ``Contest.kt`` there.
"""

from __future__ import annotations

DEFAULT_TEST_PATH_GLOBS: tuple[str, ...] = (
    # Test directories at the repo root or under any project subdirectory.
    # ``test/`` is the bats / Gradle / Maven convention; ``tests/`` is pytest's.
    "tests/**",
    "**/tests/**",
    "test/**",
    "**/test/**",
    # Gradle / Kotlin Multiplatform test source sets: src/test, src/androidTest,
    # src/commonTest, src/iosTest, src/jvmTest, ... (``src/test`` is also caught
    # by ``test/`` above; the ``*Test`` form covers every ``<target>Test`` set).
    "src/*Test/**",
    "**/src/*Test/**",
    # Python / pytest.
    "test_*.py",
    "**/test_*.py",
    "*_test.py",
    "conftest.py",
    "**/conftest.py",
    # Kotlin / JVM test files.
    "*Test.kt",
    "*Test.kts",
    # Shell (bats) and PowerShell (Pester).
    "*.bats",
    "*.Tests.ps1",
    # JavaScript / TypeScript (jest, vitest, jasmine).
    "*.test.[jt]s",
    "*.test.[jt]sx",
    "*.spec.[jt]s",
    "*.spec.[jt]sx",
)
