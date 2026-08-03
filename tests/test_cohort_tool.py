"""Tests for scripts/cohort_tool.py's Cypher query text (see docs/spec.md §2)
and its exception-handling behavior (Phase 8 hardening).

TDD: written before scripts/cohort_tool.py exists - these fail with an
ImportError until Phase 3. The query-text tests are static string-
inspection only, per docs/tasks.md Phase 2 - no live Neo4j connection, no
driver, no fakes even. Phase 8 adds fake-driver tests below covering
query_full_cohort/count_drugs_exhaustive's real exception-wrapping
behavior, which had no test coverage at all until now.

A Phase 3-followup section at the bottom of this file adds driver-injection
fault tests for query_full_cohort/count_drugs_exhaustive (get_driver()
faked via monkeypatch, never a real connection) - the rest of the file
stays static-inspection-only as described above.

Pins two module-level constants Phase 3 must define, per docs/plan.md §2's
file layout comment ("Cypher query text, exhaustive count query") and
spec.md §2's "one-clause edit of Block 5's VERIFY_PATIENTS_QUERY_TEMPLATE"
starting point:
- FULL_COHORT_QUERY_TEMPLATE: the unbounded enumeration query (Block 5's
  VERIFY_PATIENTS_QUERY_TEMPLATE with its "AND p.person_id IN $person_ids"
  clause dropped) - condition/value bound via Cypher $parameters,
  lab_property/op substituted via .format() from a fixed whitelist (not
  raw user input) same as Block 5's own precedent.
- EXHAUSTIVE_DRUG_COUNT_QUERY: the drug-count query over the full matched
  cohort, no top_k ceiling.

Security constraints under test (spec.md §2):
- condition/lab/drug_a/drug_b must never be interpolated into the query
  text itself via f-string or .format() - only passed as Cypher
  $parameters at execution time.
- Read-only: MATCH/RETURN only, no CREATE/MERGE/DELETE/SET.
"""
import inspect
import re

import pytest
from neo4j.exceptions import Neo4jError

from scripts import cohort_tool
from scripts.cohort_tool import (
    EXHAUSTIVE_DRUG_COUNT_QUERY,
    FULL_COHORT_QUERY_TEMPLATE,
    CohortServiceError,
    count_drugs_exhaustive,
    query_full_cohort,
)

_FORBIDDEN_WRITE_KEYWORDS = ["CREATE", "MERGE", "DELETE", "SET"]
# The exact field names spec.md §2 says must never be string-interpolated
# into the query text - if any of these ever appears in the source as an
# f-string/.format() placeholder (e.g. "{condition}"), that's the
# vulnerability this test exists to catch.
_MUST_NEVER_BE_TEMPLATE_PLACEHOLDERS = ["condition", "lab", "drug_a", "drug_b"]


def _assert_read_only(query_text: str) -> None:
    upper = query_text.upper()
    assert "MATCH" in upper or "RETURN" in upper, "expected a MATCH/RETURN query"
    for keyword in _FORBIDDEN_WRITE_KEYWORDS:
        assert not re.search(rf"\b{keyword}\b", upper), f"query must not contain {keyword}"


def test_full_cohort_query_is_read_only():
    _assert_read_only(FULL_COHORT_QUERY_TEMPLATE)


def test_exhaustive_drug_count_query_is_read_only():
    _assert_read_only(EXHAUSTIVE_DRUG_COUNT_QUERY)


def test_full_cohort_query_binds_condition_as_a_cypher_parameter():
    assert "$condition" in FULL_COHORT_QUERY_TEMPLATE


def test_full_cohort_query_binds_value_as_a_cypher_parameter():
    assert "$value" in FULL_COHORT_QUERY_TEMPLATE


def test_full_cohort_query_drops_the_person_id_prefilter():
    # spec.md §2: this is Block 5's VERIFY_PATIENTS_QUERY_TEMPLATE with just
    # the trailing "AND p.person_id IN $person_ids" clause removed - the
    # whole point of Role 2 is no top_k ceiling on the candidate population.
    assert "person_ids" not in FULL_COHORT_QUERY_TEMPLATE


def test_full_cohort_query_reuses_block5_relationship_and_property_names():
    assert "HAS_CONDITION" in FULL_COHORT_QUERY_TEMPLATE
    assert "condition_name" in FULL_COHORT_QUERY_TEMPLATE


def test_exhaustive_drug_count_query_reuses_block5_relationship_name():
    assert "PRESCRIBED" in EXHAUSTIVE_DRUG_COUNT_QUERY


def test_no_query_constant_contains_an_fstring_or_format_placeholder_for_the_protected_fields():
    for query_text in (FULL_COHORT_QUERY_TEMPLATE, EXHAUSTIVE_DRUG_COUNT_QUERY):
        for field in _MUST_NEVER_BE_TEMPLATE_PLACEHOLDERS:
            assert "{" + field + "}" not in query_text, (
                f"{field} must never be an f-string/.format() placeholder in the "
                "query text - it must only ever be bound as a Cypher $parameter"
            )


def test_module_source_never_builds_a_query_by_interpolating_the_protected_fields():
    # Backstop beyond the two constants above: scans the whole module's
    # source for the two Python-level interpolation mechanisms (f-strings
    # and .format()) applied to any of the protected field names, in case
    # a future query is built some other way than the two constants this
    # test already names.
    source = inspect.getsource(cohort_tool)
    for field in _MUST_NEVER_BE_TEMPLATE_PLACEHOLDERS:
        assert f"{{{field}}}" not in source, (
            f"found an f-string/.format() placeholder for {field} in "
            "scripts/cohort_tool.py's source"
        )
        assert not re.search(rf"\.format\([^)]*\b{field}\s*=", source), (
            f".format({field}=...) found in scripts/cohort_tool.py's source"
        )


# --- exception-handling behavior (Phase 8 hardening) -----------------------


class _FakeResult:
    def __init__(self, single_value=None, rows=None):
        self._single_value = single_value
        self._rows = rows or []

    def single(self):
        return self._single_value

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, raise_exc=None, single_value=None, rows=None, recorded_queries=None):
        self._raise_exc = raise_exc
        self._single_value = single_value
        self._rows = rows
        self._recorded_queries = recorded_queries

    def run(self, query, **params):
        if self._recorded_queries is not None:
            self._recorded_queries.append(query)
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeResult(single_value=self._single_value, rows=self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeDriver:
    def __init__(self, raise_exc=None, single_value=None, rows=None):
        self._raise_exc = raise_exc
        self._single_value = single_value
        self._rows = rows
        # Populated with each Query object session.run() was called with,
        # so tests can inspect e.g. .timeout without a live driver.
        self.recorded_queries = []

    def session(self, database=None):
        return _FakeSession(
            raise_exc=self._raise_exc,
            single_value=self._single_value,
            rows=self._rows,
            recorded_queries=self.recorded_queries,
        )


def test_query_full_cohort_wraps_a_connection_error_as_retryable_with_the_real_message():
    driver = _FakeDriver(raise_exc=ConnectionError("connection refused"))

    try:
        query_full_cohort("Essential hypertension", "SBP", "above", 140, driver=driver)
        assert False, "expected CohortServiceError"
    except CohortServiceError as exc:
        # The real message reaches the caveat text, not just the
        # exception's type name.
        assert exc.detail == "connection refused"
        assert exc.retryable is True


def test_query_full_cohort_wraps_an_unknown_exception_as_non_retryable():
    driver = _FakeDriver(raise_exc=RuntimeError("unexpected bug"))

    try:
        query_full_cohort("Essential hypertension", "SBP", "above", 140, driver=driver)
        assert False, "expected CohortServiceError"
    except CohortServiceError as exc:
        assert exc.detail == "unexpected bug"
        # classify_exception has no case for a bare RuntimeError - it's
        # "unknown", meaning a real bug or bad input, not transient infra.
        # Retrying 3 times wouldn't fix it, so this must not be retryable.
        assert exc.retryable is False


def test_query_full_cohort_wraps_a_real_transaction_timeout_as_retryable():
    # Regression test for a PR #8 review's finding: a real Query(timeout=...)
    # expiring server-side raises a real neo4j ClientError carrying the
    # Neo.ClientError.Transaction.TransactionTimedOut code - not a
    # ServiceUnavailable, and not one of the generic ConnectionError/
    # RuntimeError fakes the tests above already cover. Before
    # classify_exception recognized this code, this exact scenario fell
    # through to "unknown" and was wrongly marked non-retryable.
    timeout_exc = Neo4jError._hydrate_neo4j(
        code="Neo.ClientError.Transaction.TransactionTimedOut",
        message="The transaction has been terminated",
    )
    driver = _FakeDriver(raise_exc=timeout_exc)

    try:
        query_full_cohort("Essential hypertension", "SBP", "above", 140, driver=driver)
        assert False, "expected CohortServiceError"
    except CohortServiceError as exc:
        assert exc.retryable is True


def test_count_drugs_exhaustive_wraps_a_connection_error_as_retryable_with_the_real_message():
    driver = _FakeDriver(raise_exc=ConnectionError("connection refused"))

    try:
        count_drugs_exhaustive([1, 2, 3], "Lisinopril", "Amlodipine", driver=driver)
        assert False, "expected CohortServiceError"
    except CohortServiceError as exc:
        assert exc.detail == "connection refused"
        assert exc.retryable is True


def test_count_drugs_exhaustive_wraps_an_unknown_exception_as_non_retryable():
    driver = _FakeDriver(raise_exc=RuntimeError("unexpected bug"))

    try:
        count_drugs_exhaustive([1, 2, 3], "Lisinopril", "Amlodipine", driver=driver)
        assert False, "expected CohortServiceError"
    except CohortServiceError as exc:
        assert exc.detail == "unexpected bug"
        assert exc.retryable is False


# --- injectable GRAPH_QUERY_TIMEOUT (Phase 8 hardening) ---------------------


def test_query_full_cohort_defaults_to_the_module_level_timeout():
    driver = _FakeDriver(single_value={"matched_ids": []})

    query_full_cohort("Essential hypertension", "SBP", "above", 140, driver=driver)

    assert driver.recorded_queries[0].timeout == cohort_tool.GRAPH_QUERY_TIMEOUT


def test_query_full_cohort_honors_a_graph_query_timeout_override():
    driver = _FakeDriver(single_value={"matched_ids": []})

    query_full_cohort(
        "Essential hypertension", "SBP", "above", 140, driver=driver, graph_query_timeout=3
    )

    assert driver.recorded_queries[0].timeout == 3


def test_count_drugs_exhaustive_defaults_to_the_module_level_timeout():
    driver = _FakeDriver(rows=[{"drug": "Lisinopril", "patient_count": 1}])

    count_drugs_exhaustive([1], "Lisinopril", "Amlodipine", driver=driver)

    assert driver.recorded_queries[0].timeout == cohort_tool.GRAPH_QUERY_TIMEOUT


def test_count_drugs_exhaustive_honors_a_graph_query_timeout_override():
    driver = _FakeDriver(rows=[{"drug": "Lisinopril", "patient_count": 1}])

    count_drugs_exhaustive([1], "Lisinopril", "Amlodipine", driver=driver, graph_query_timeout=3)

    assert driver.recorded_queries[0].timeout == 3


# --- get_driver() failures must be caught, not raised raw ---------------


def _raising_get_driver(exc):
    def _fn():
        raise exc

    return _fn


def test_query_full_cohort_wraps_a_get_driver_failure_as_cohort_service_error(monkeypatch):
    # get_driver() must be called inside the try block, not before it - a
    # connection failure at driver-construction time is exactly as much a
    # tool failure as one during the query itself, and must degrade the
    # same way (CohortServiceError), not propagate the raw exception.
    monkeypatch.setattr(cohort_tool, "get_driver", _raising_get_driver(RuntimeError("no db")))

    with pytest.raises(CohortServiceError):
        query_full_cohort("hypertension", "SBP", "above", 140)


def test_count_drugs_exhaustive_wraps_a_get_driver_failure_as_cohort_service_error(monkeypatch):
    monkeypatch.setattr(cohort_tool, "get_driver", _raising_get_driver(RuntimeError("no db")))

    with pytest.raises(CohortServiceError):
        count_drugs_exhaustive([1, 2, 3], "Lisinopril", "Amlodipine")
