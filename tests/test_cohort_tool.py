"""Tests for scripts/cohort_tool.py's Cypher query text (see docs/spec.md §2).

TDD: written before scripts/cohort_tool.py exists - these fail with an
ImportError until Phase 3. Static string-inspection only, per docs/tasks.md
Phase 2 - no live Neo4j connection, no driver, no fakes even.

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

from scripts import cohort_tool
from scripts.cohort_tool import EXHAUSTIVE_DRUG_COUNT_QUERY, FULL_COHORT_QUERY_TEMPLATE

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
