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

import pytest

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


def test_connectex_is_transient_for_gh() -> None:
    # Windows Winsock's transport error text, surfaced through Go's own
    # syscall-level wrapping -- gh's net/http backend on Windows, not git:
    # git never emits this token (see the module docstring and finding 1 of
    # the git-retry-primitive review). Real shape from this fleet's own
    # events.db (a gh-side main_ci_reclaim_failed row).
    assert (
        is_transient_network_error(
            "dial tcp 172.182.252.137:443: connectex: No connection could be made "
            "because the target machine actively refused it."
        )
        is True
    )


# Real git/libcurl/OpenSSL/schannel/Winsock transport error text, most pinned
# against this host's own real `git fetch` against an unroutable remote
# (`git fetch origin main` with the remote pointed at 127.0.0.1:1) or the
# libcurl/schannel/OpenSSL docs' own verbatim wording. Before the fix these
# 8 of 10 realistic shapes classified as terminal (False) even though every
# one of them is the exact class of transient blip this module exists to
# retry -- a classifier ported verbatim from gh's Go-idiom text was inert for
# git. See transient_errors.py's module docstring.
_REAL_GIT_TRANSIENT_STDERRS = [
    pytest.param(
        "fatal: unable to access 'https://127.0.0.1:1/x/y.git/': Failed to connect to "
        "127.0.0.1 port 1 after 2093 ms: Couldn't connect to server",
        id="couldnt_connect_to_server",
    ),
    pytest.param(
        "OpenSSL SSL_read: Connection was reset, errno 10054",
        id="ssl_read_connection_was_reset",
    ),
    pytest.param(
        "error: RPC failed; curl 56 Recv failure: Connection was reset",
        id="recv_failure_connection_reset",
    ),
    pytest.param(
        "The requested URL returned error: 502",
        id="requested_url_returned_error_502",
    ),
    pytest.param(
        "Operation timed out after 120000 milliseconds with 0 bytes received",
        id="operation_timed_out_after",
    ),
    pytest.param(
        "schannel: failed to receive handshake, SSL/TLS connection failed",
        id="schannel_handshake_failed",
    ),
    pytest.param(
        "fatal: the remote end hung up unexpectedly",
        id="remote_end_hung_up",
    ),
    pytest.param(
        "fatal: unable to access 'https://github.com/x/y.git/': Could not resolve "
        "host: github.com",
        id="could_not_resolve_host",
    ),
]


@pytest.mark.parametrize("stderr", _REAL_GIT_TRANSIENT_STDERRS)
def test_real_git_transport_shape_is_transient(stderr: str) -> None:
    assert is_transient_network_error(stderr) is True


# Additional libcurl/OpenSSL/GnuTLS shapes documented by those tools' own
# error text (not in the review's positive-control table above, but part of
# the same required-fix pattern list) -- send-side mirror of recv failure,
# the SSL_connect (as opposed to SSL_read) OpenSSL phase, curl's error 52,
# and a GnuTLS-backed libcurl build's handshake failure.
@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param(
            "error: RPC failed; curl 55 Send failure: Connection was reset",
            id="send_failure",
        ),
        pytest.param(
            "OpenSSL SSL_connect: Connection was reset, errno 10054",
            id="ssl_connect",
        ),
        pytest.param(
            "The requested URL returned error: 429",
            id="requested_url_returned_error_429",
        ),
        pytest.param(
            "curl: (52) Empty reply from server",
            id="empty_reply_from_server",
        ),
        pytest.param(
            "gnutls_handshake() failed: Error in the pull function",
            id="gnutls_handshake_failed",
        ),
    ],
)
def test_additional_libcurl_shapes_are_transient(stderr: str) -> None:
    assert is_transient_network_error(stderr) is True


def test_generic_exit_code_text_is_not_itself_transient() -> None:
    # The shadowed generic `.error` string `run_captured` always sets on a
    # non-zero exit (issue #1777 finding 3) carries no transport information
    # at all -- it must never be misread as a transient shape by accident.
    assert is_transient_network_error("command exited 128") is False


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
