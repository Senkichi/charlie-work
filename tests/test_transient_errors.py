"""Tests for the shared transient-network-error classifier.

``transient_errors.is_transient_network_error`` is the single definition of
"transient network failure" shared by ``GitHub.run()`` (via
``github._is_transient_gh_error``, now a thin alias) and
``git_retry.run_git_with_retry``. These tests pin its behavior for both the
git-specific transport shapes this task adds and the terminal cases that must
never be retried, plus an identity guard so the ``github.py`` alias cannot
silently diverge back into a second copy of the allowlist.
"""

from __future__ import annotations

import charlie_work.github as github_module
from charlie_work.transient_errors import is_transient_network_error


def test_github_alias_is_the_shared_classifier() -> None:
    # A regression guard, not a behavioral test: if a future edit reintroduces
    # a second, independently-maintained allowlist in github.py, this fails
    # immediately rather than letting the two drift apart silently.
    assert github_module._is_transient_gh_error is is_transient_network_error


def test_connection_reset_is_transient() -> None:
    assert is_transient_network_error("fatal: connection reset by peer") is True


def test_tls_handshake_timeout_is_transient() -> None:
    assert is_transient_network_error("Post ...: net/http: TLS handshake timeout") is True


def test_could_not_resolve_host_is_transient() -> None:
    # git/libcurl's own transport error text for a DNS blip -- absent from
    # gh's Go-idiom output, so this is a git-specific addition this task made.
    assert (
        is_transient_network_error(
            "fatal: unable to access 'https://github.com/x/y.git/': "
            "Could not resolve host: github.com"
        )
        is True
    )


def test_connectex_is_transient() -> None:
    # Windows Winsock's own transport error text, likewise git-specific.
    assert (
        is_transient_network_error(
            "fatal: unable to access 'https://github.com/x/y.git/': "
            "Failed to connect to github.com port 443: Recv failure: Connection was reset"
            " (connectex)"
        )
        is True
    )


def test_http_5xx_is_transient() -> None:
    assert is_transient_network_error("HTTP 502: Bad Gateway") is True


def test_i_o_timeout_is_transient() -> None:
    assert is_transient_network_error("dial tcp 140.82.113.3:443: i/o timeout") is True


def test_non_fast_forward_is_terminal() -> None:
    # A real ``git pull --ff-only`` rejection: must never be retried, since
    # retrying a non-FF pull can never succeed and masks a real divergence.
    assert is_transient_network_error("fatal: Not possible to fast-forward, aborting.") is False


def test_merge_conflict_is_terminal() -> None:
    assert (
        is_transient_network_error(
            "CONFLICT (content): Merge conflict in src/charlie_work/state.py"
        )
        is False
    )


def test_bad_credentials_is_terminal() -> None:
    assert is_transient_network_error("fatal: Authentication failed: Bad credentials") is False


def test_repository_not_found_is_terminal() -> None:
    assert (
        is_transient_network_error(
            "remote: Repository not found.\n"
            "fatal: repository 'https://github.com/x/y.git/' not found"
        )
        is False
    )


def test_http_401_is_terminal() -> None:
    assert is_transient_network_error("gh: HTTP 401: Bad credentials") is False


def test_http_403_rate_limit_is_transient() -> None:
    # A 403 is terminal by default, but a rate-limit 403 is the one carve-out.
    assert is_transient_network_error("gh: HTTP 403: API rate limit exceeded") is True


def test_http_403_non_rate_limit_is_terminal() -> None:
    assert is_transient_network_error("gh: HTTP 403: Resource not accessible") is False


def test_unknown_error_defaults_terminal() -> None:
    assert is_transient_network_error("fatal: something completely unrecognized") is False


def test_classification_is_case_insensitive() -> None:
    assert is_transient_network_error("FATAL: CONNECTION RESET BY PEER") is True
