"""``load_config`` validation of the ``test_adequacy`` and
``coverage_probe`` sections: tuple-field coercion, bad-type rejection,
and full-override acceptance.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import load_config


def test_config_test_adequacy_coerces_tuple_fields_to_tuple(tmp_path: Path) -> None:
    """YAML lists in test_adequacy tuple fields round-trip to tuples."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """test_adequacy:
  test_path_globs:
    - "tests/**"
    - "test_*.py"
  exempt_path_globs:
    - "*.md"
    - "docs/**"
  assertion_markers:
    - "assert "
    - "pytest.raises"
  comment_prefixes:
    - "#"
    - "//"
  stub_test_seam_keywords:
    - "route"
    - "call_model"
  coverage_command:
    - "pytest"
    - "--cov"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config.test_adequacy.test_path_globs, tuple)
    assert config.test_adequacy.test_path_globs == ("tests/**", "test_*.py")
    assert isinstance(config.test_adequacy.exempt_path_globs, tuple)
    assert config.test_adequacy.exempt_path_globs == ("*.md", "docs/**")
    assert isinstance(config.test_adequacy.assertion_markers, tuple)
    assert config.test_adequacy.assertion_markers == ("assert ", "pytest.raises")
    assert isinstance(config.test_adequacy.comment_prefixes, tuple)
    assert config.test_adequacy.comment_prefixes == ("#", "//")
    assert isinstance(config.test_adequacy.stub_test_seam_keywords, tuple)
    assert config.test_adequacy.stub_test_seam_keywords == ("route", "call_model")
    assert isinstance(config.test_adequacy.coverage_command, tuple)
    assert config.test_adequacy.coverage_command == ("pytest", "--cov")


def test_config_rejects_non_list_test_adequacy_tuple_field(tmp_path: Path) -> None:
    """Tuple fields given as scalars raise ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  test_path_globs: tests/**\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for non-list tuple field")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for non-list tuple field")

    assert "test_adequacy" in message
    assert "test_path_globs" in message
    assert "must be a list" in message


def test_config_rejects_non_str_element_in_test_adequacy_tuple_field(tmp_path: Path) -> None:
    """Tuple fields with non-str elements raise ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  test_path_globs:\n    - tests/**\n    - 123\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for non-str element")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for non-str element")

    assert "test_adequacy" in message
    assert "test_path_globs" in message
    assert "element of type" in message


def test_config_rejects_bad_type_min_product_lines(tmp_path: Path) -> None:
    """min_product_lines as string raises ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  min_product_lines: ten\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for bad type min_product_lines")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bad type min_product_lines")

    assert "test_adequacy" in message
    assert "min_product_lines" in message
    assert "must be an int" in message


def test_config_rejects_bad_type_min_diff_coverage(tmp_path: Path) -> None:
    """min_diff_coverage as string raises ConfigError; int is accepted."""
    from charlie_work.config import ConfigError

    # Reject string
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  min_diff_coverage: high\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for bad type min_diff_coverage")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bad type min_diff_coverage")

    assert "test_adequacy" in message
    assert "min_diff_coverage" in message
    assert "must be a float" in message

    # Accept int
    config_path.write_text(
        "test_adequacy:\n  min_diff_coverage: 1\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.test_adequacy.min_diff_coverage == 1


def test_config_rejects_empty_exempt_marker(tmp_path: Path) -> None:
    """exempt_marker as empty string raises ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'test_adequacy:\n  exempt_marker: ""\n',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for empty exempt_marker")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for empty exempt_marker")

    assert "test_adequacy" in message
    assert "exempt_marker" in message
    assert "non-empty string" in message


def test_config_rejects_non_bool_test_adequacy_flags(tmp_path: Path) -> None:
    """Boolean flags as strings raise ConfigError."""
    from charlie_work.config import ConfigError

    for bool_key in ("enabled", "coverage_enabled", "require_assertions"):
        config_path = tmp_path / "orchestrator.config.yaml"
        config_path.write_text(
            f'test_adequacy:\n  {bool_key}: "true"\n',
            encoding="utf-8",
        )

        try:
            load_config(config_path)
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")
        except ConfigError as exc:
            message = str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")

        assert "test_adequacy" in message
        assert bool_key in message
        assert "must be a bool" in message


def test_config_accepts_full_test_adequacy_override(tmp_path: Path) -> None:
    """A YAML block overriding every field loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """test_adequacy:
  enabled: true
  min_product_lines: 20
  test_path_globs:
    - "custom_tests/**"
  exempt_path_globs:
    - "*.txt"
  assertion_markers:
    - "custom_assert"
  comment_prefixes:
    - "//"
  require_assertions: true
  stub_test_seam_keywords:
    - "route"
    - "byte"
  exempt_marker: "Custom-exempt:"
  coverage_enabled: true
  coverage_command:
    - "custom"
    - "cov"
  min_diff_coverage: 0.5
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.test_adequacy.enabled is True
    assert config.test_adequacy.min_product_lines == 20
    assert config.test_adequacy.test_path_globs == ("custom_tests/**",)
    assert config.test_adequacy.exempt_path_globs == ("*.txt",)
    assert config.test_adequacy.assertion_markers == ("custom_assert",)
    assert config.test_adequacy.comment_prefixes == ("//",)
    assert config.test_adequacy.require_assertions is True
    assert config.test_adequacy.stub_test_seam_keywords == ("route", "byte")
    assert config.test_adequacy.exempt_marker == "Custom-exempt:"
    assert config.test_adequacy.coverage_enabled is True
    assert config.test_adequacy.coverage_command == ("custom", "cov")
    assert config.test_adequacy.min_diff_coverage == 0.5


def test_config_coverage_probe_coerces_tuple_fields_to_tuple(tmp_path: Path) -> None:
    """YAML lists in coverage_probe tuple fields round-trip to tuples."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """coverage_probe:
  enabled: true
  test_path_globs:
    - "tests/**"
    - "test_*.py"
  exempt_path_globs:
    - "*.md"
  comment_prefixes:
    - "#"
    - "//"
  branch_tokens:
    - "if "
    - "elif "
  assertion_markers:
    - "assert "
    - "pytest.raises"
  branch_to_assert_ratio_threshold: 3.5
  check_unwired_symbols: false
  test_function_prefix: "def check_"
  private_name_prefix: "__"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.coverage_probe.enabled is True
    assert isinstance(config.coverage_probe.test_path_globs, tuple)
    assert config.coverage_probe.test_path_globs == ("tests/**", "test_*.py")
    assert config.coverage_probe.exempt_path_globs == ("*.md",)
    assert config.coverage_probe.comment_prefixes == ("#", "//")
    assert config.coverage_probe.branch_tokens == ("if ", "elif ")
    assert config.coverage_probe.assertion_markers == ("assert ", "pytest.raises")
    assert config.coverage_probe.branch_to_assert_ratio_threshold == 3.5
    assert config.coverage_probe.check_unwired_symbols is False
    assert config.coverage_probe.test_function_prefix == "def check_"
    assert config.coverage_probe.private_name_prefix == "__"


def test_config_rejects_non_list_coverage_probe_tuple_field(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "coverage_probe:\n  branch_tokens: 'if '\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for non-list tuple field")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for non-list tuple field")

    assert "coverage_probe" in message
    assert "branch_tokens" in message
    assert "must be a list" in message


def test_config_rejects_bad_type_branch_to_assert_ratio_threshold(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "coverage_probe:\n  branch_to_assert_ratio_threshold: high\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for bad type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bad type")

    assert "coverage_probe" in message
    assert "branch_to_assert_ratio_threshold" in message
    assert "must be a float" in message


def test_config_rejects_non_bool_coverage_probe_flags(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    for bool_key in ("enabled", "check_unwired_symbols"):
        config_path = tmp_path / "orchestrator.config.yaml"
        config_path.write_text(
            f'coverage_probe:\n  {bool_key}: "true"\n',
            encoding="utf-8",
        )

        try:
            load_config(config_path)
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")
        except ConfigError as exc:
            message = str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")

        assert "coverage_probe" in message
        assert bool_key in message
        assert "must be a bool" in message


def test_config_accepts_full_coverage_probe_override(tmp_path: Path) -> None:
    """A YAML block overriding every field loads correctly and does NOT
    disturb TestAdequacyConfig's reserved Tier-3 fields (independent
    sections -- issues #1260/#1261 design item 2)."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """coverage_probe:
  enabled: true
  test_path_globs:
    - "custom_tests/**"
  exempt_path_globs:
    - "*.txt"
  comment_prefixes:
    - "//"
  branch_tokens:
    - "if "
  assertion_markers:
    - "custom_assert"
  branch_to_assert_ratio_threshold: 2.0
  check_unwired_symbols: false
  test_function_prefix: "def scenario_"
  private_name_prefix: "__"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.coverage_probe.enabled is True
    assert config.coverage_probe.test_path_globs == ("custom_tests/**",)
    assert config.coverage_probe.exempt_path_globs == ("*.txt",)
    assert config.coverage_probe.comment_prefixes == ("//",)
    assert config.coverage_probe.branch_tokens == ("if ",)
    assert config.coverage_probe.assertion_markers == ("custom_assert",)
    assert config.coverage_probe.branch_to_assert_ratio_threshold == 2.0
    assert config.coverage_probe.check_unwired_symbols is False
    assert config.coverage_probe.test_function_prefix == "def scenario_"
    assert config.coverage_probe.private_name_prefix == "__"
    # TestAdequacyConfig's reserved Tier-3 fields are untouched defaults.
    assert config.test_adequacy.coverage_enabled is False
    assert config.test_adequacy.min_diff_coverage == 0.0
