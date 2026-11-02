"""Request values: mutation-ness is derived, never listed (ADR-0006)."""

from __future__ import annotations

import pytest

from charlie_work.github_transport import CliCommand, CliRequest, GraphQLRequest, RestRequest


@pytest.mark.parametrize(
    ("method", "mutation"),
    [("GET", False), ("POST", True), ("PUT", True), ("PATCH", True), ("DELETE", True)],
)
def test_rest_mutation_is_derived_from_the_method(method: str, mutation: bool) -> None:
    body = None if method == "GET" else {"a": 1}
    assert RestRequest.of(method, "repos/o/r/x", body=body).is_mutation is mutation  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("document", "operation"),
    [
        ("query { viewer { login } }", "query"),
        ("{ viewer { login } }", "query"),
        (
            "mutation M($i: ID!) { closeIssue(input: {issueId: $i}) { clientMutationId } }",
            "mutation",
        ),
        ("subscription S { x }", "subscription"),
        ("# mutation in a comment\nquery Q { a }", "query"),
        ('query Q { a(s: "mutation { x }") }', "query"),
        ('query Q { a(s: """mutation { x }""") }', "query"),
        ("fragment F on User { login }\nquery Q { viewer { ...F } }", "query"),
        ('query Q($m: String = "mutation") { a }', "query"),
        ("\ufeffmutation M { a }", "mutation"),
    ],
)
def test_graphql_operation_type_is_lexed_not_substring_matched(
    document: str, operation: str
) -> None:
    request = GraphQLRequest.of(document)
    assert request.operation == operation
    assert request.is_mutation is (operation != "query")  # subscription fails closed


@pytest.mark.parametrize(
    "document",
    ["", "fragment F on U { a }", "query A { a } query B { b }", "query A { a } mutation B { b }"],
)
def test_a_document_without_exactly_one_operation_is_rejected_at_construction(
    document: str,
) -> None:
    with pytest.raises(ValueError):
        GraphQLRequest.of(document)


def test_graphql_variables_must_be_an_object() -> None:
    with pytest.raises(ValueError):
        GraphQLRequest("query { a }", "[1]")


def test_equal_payloads_are_equal_and_hashable_regardless_of_key_order() -> None:
    a = RestRequest.of("POST", "repos/o/r/issues", body={"b": 1, "a": 2})
    b = RestRequest.of("POST", "repos/o/r/issues", body='{"a": 2, "b": 1}')
    assert a == b and hash(a) == hash(b)
    g1 = GraphQLRequest.of("query { a }", {"x": 1, "y": 2})
    g2 = GraphQLRequest.of("query { a }", {"y": 2, "x": 1})
    assert g1 == g2 and len({g1, g2}) == 1


def test_a_get_cannot_carry_a_body_and_a_route_cannot_carry_a_query() -> None:
    with pytest.raises(ValueError):
        RestRequest.of("GET", "repos/o/r/x", body={"a": 1})
    with pytest.raises(ValueError):
        RestRequest.of("GET", "repos/o/r/x?y=1")
    with pytest.raises(ValueError):
        RestRequest("TRACE", "x")  # type: ignore[arg-type]


def test_placeholders_resolve_and_reject_unsafe_owner_repo() -> None:
    request = RestRequest.of("GET", "/repos/{owner}/{repo}/pulls", query={"state": "open"})
    assert request.needs_repo
    resolved = request.resolve("octo", "hello")
    assert resolved.route == "repos/octo/hello/pulls"
    assert not resolved.needs_repo
    assert resolved.target() == "repos/octo/hello/pulls?state=open"
    with pytest.raises(ValueError):
        request.resolve("octo/evil", "hello")
    with pytest.raises(ValueError):
        request.resolve("", "hello")


def test_cli_requests_are_never_mutations() -> None:
    assert not CliRequest(CliCommand.AUTH_TOKEN).is_mutation
    assert CliRequest(CliCommand.AUTH_STATUS).describe() == "gh auth status"
