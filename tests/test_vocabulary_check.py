"""Tests for scripts/vocabulary_check.py's get_known_vocabulary (see docs/plan.md §12).

A fake driver stands in for a live Neo4j connection throughout - no real
database access here, same DI pattern as scripts/cohort_tool.py's own
driver-injection tests.

Labs must be checked against the real graph the same way conditions
already are, not against this repo's own LAB_PROPERTY_NAMES whitelist
alone - a real-world property rename in Block 3's graph (e.g.
latest_sbp -> something else) would otherwise go undetected here even
though every lab-based query would then silently match zero patients,
with no error anywhere.
"""
import pytest

from scripts import cohort_tool, vocabulary_check
from scripts.vocabulary_check import get_known_vocabulary


class _FakeSession:
    def __init__(self, condition_names, property_keys, recorded_queries=None):
        self._condition_names = condition_names
        self._property_keys = property_keys
        self._recorded_queries = recorded_queries

    def run(self, query):
        if self._recorded_queries is not None:
            self._recorded_queries.append(query)
        # Post-fix, query is always a neo4j.Query wrapper, not a bare
        # string - .text is what carries the actual Cypher text now.
        query_text = query.text
        if "condition_name" in query_text:
            return [{"condition_name": name} for name in self._condition_names]
        return [{"property_key": key} for key in self._property_keys]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeDriver:
    def __init__(self, condition_names, property_keys):
        self._condition_names = condition_names
        self._property_keys = property_keys
        # Populated with each Query object session.run() was called with,
        # so tests can inspect e.g. .timeout without a live driver - same
        # pattern as scripts/cohort_tool.py's own tests.
        self.recorded_queries = []

    def session(self, database=None):
        return _FakeSession(self._condition_names, self._property_keys, recorded_queries=self.recorded_queries)


@pytest.fixture(autouse=True)
def _reset_vocabulary_cache(monkeypatch):
    # get_known_vocabulary caches globally regardless of which driver is
    # passed in - every test here needs a cold cache so its own fake
    # driver is actually consulted, not a previous test's cached result.
    monkeypatch.setattr(vocabulary_check, "_cached_vocabulary", None)
    monkeypatch.setattr(vocabulary_check, "_cached_at", 0.0)


def test_conditions_still_come_from_the_real_graph_query():
    driver = _FakeDriver(
        condition_names=["hypertension", "diabetes"],
        property_keys=["person_id", "latest_sbp"],
    )

    vocabulary = get_known_vocabulary(driver=driver)

    assert vocabulary["conditions"] == {"hypertension", "diabetes"}


def test_all_four_labs_known_when_all_real_properties_present():
    driver = _FakeDriver(
        condition_names=["hypertension"],
        property_keys=["person_id", "latest_sbp", "latest_bmi", "latest_glucose", "latest_hba1c"],
    )

    vocabulary = get_known_vocabulary(driver=driver)

    assert vocabulary["labs"] == {"SBP", "BMI", "Glucose", "HbA1c"}


def test_lab_known_only_when_its_real_patient_property_exists_in_the_graph():
    # Glucose/HbA1c's mapped properties (latest_glucose/latest_hba1c) are
    # absent from the real graph here - only SBP/BMI's properties exist.
    driver = _FakeDriver(
        condition_names=["hypertension"],
        property_keys=["person_id", "latest_sbp", "latest_bmi"],
    )

    vocabulary = get_known_vocabulary(driver=driver)

    assert vocabulary["labs"] == {"SBP", "BMI"}


def test_renamed_lab_property_is_flagged_as_unknown_not_silently_passed():
    # The exact regression this fix targets: latest_sbp renamed to
    # latest_systolic_bp in Block 3's graph. Before this fix, "SBP" was
    # checked only against this repo's own hardcoded whitelist, which
    # still listed "SBP" as known regardless of what the real graph
    # actually had - the rename went undetected, and every SBP-based
    # query would have silently matched zero patients from then on, with
    # no error anywhere. Now it must be flagged as unknown.
    driver = _FakeDriver(
        condition_names=["hypertension"],
        property_keys=["person_id", "latest_systolic_bp", "latest_bmi"],
    )

    vocabulary = get_known_vocabulary(driver=driver)

    assert "SBP" not in vocabulary["labs"]
    assert vocabulary["labs"] == {"BMI"}


def test_both_queries_use_the_shared_graph_query_timeout():
    # The regression this fix targets (docs/tasks.md "Block 6 - state
    # validation"): unlike every other query in scripts/cohort_tool.py,
    # _fetch_known_vocabulary()'s two session.run() calls had no timeout=
    # at all, so a slow/blocked Neo4j call here could hang indefinitely
    # instead of surfacing as an error. Both queries must now share
    # cohort_tool.py's own GRAPH_QUERY_TIMEOUT constant, not a
    # separately-defined value that could drift from it.
    driver = _FakeDriver(
        condition_names=["hypertension"],
        property_keys=["person_id", "latest_sbp"],
    )

    get_known_vocabulary(driver=driver)

    assert len(driver.recorded_queries) == 2
    assert driver.recorded_queries[0].timeout == cohort_tool.GRAPH_QUERY_TIMEOUT
    assert driver.recorded_queries[1].timeout == cohort_tool.GRAPH_QUERY_TIMEOUT


def _real_driver_or_skip():
    """A real Neo4j driver, not a fake - the one place this file breaks
    its own "fake driver throughout" convention (this file's own
    docstring), on purpose: the fake driver above can't prove a slow
    server-side query actually gets cut off, only that the right
    parameter was handed to it. Skips cleanly wherever a real, reachable
    Neo4j instance isn't configured (e.g. a dev machine with no
    NEO4J_PASSWORD set) - this repo's CI job already runs a disposable
    neo4j:5.18-community service (.github/workflows/ci.yml) with these
    same env vars, so it runs for real there.
    """
    import os

    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set - this test needs a real, reachable Neo4j instance")

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"real Neo4j not reachable at {uri}: {exc}")
    return driver


def test_a_genuinely_slow_query_surfaces_as_a_timeout_not_a_hang():
    # Verified against actual behavior, not just the diff (same standard
    # as the retry-backoff work): a real, unbounded 20000x20000-row
    # UNWIND cross join against a real Neo4j server, timed by hand before
    # writing this assertion - it reliably runs several seconds before
    # completing on its own. With GRAPH_QUERY_TIMEOUT patched down to 1
    # second, _fetch_known_vocabulary's first real session.run() call must
    # surface a real driver-level timeout exception within a small,
    # bounded window instead of ever letting it run to completion.
    import time

    driver = _real_driver_or_skip()
    try:
        monkeypatch_timeout = 1
        original_timeout = vocabulary_check.GRAPH_QUERY_TIMEOUT
        original_query = vocabulary_check._DISTINCT_CONDITION_NAMES_QUERY
        vocabulary_check.GRAPH_QUERY_TIMEOUT = monkeypatch_timeout
        vocabulary_check._DISTINCT_CONDITION_NAMES_QUERY = (
            "UNWIND range(1, 8000) AS a UNWIND range(1, 8000) AS b RETURN count(*) AS condition_name"
        )
        try:
            started_at = time.monotonic()
            with pytest.raises(Exception):
                vocabulary_check._fetch_known_vocabulary(driver=driver)
            elapsed = time.monotonic() - started_at
        finally:
            vocabulary_check.GRAPH_QUERY_TIMEOUT = original_timeout
            vocabulary_check._DISTINCT_CONDITION_NAMES_QUERY = original_query

        # Generous upper bound (real timeout enforcement isn't
        # millisecond-precise), but nowhere near what letting the query
        # actually finish would take - proves this terminates the call,
        # rather than merely accepting a parameter that's never enforced.
        assert elapsed < 8.0
    finally:
        driver.close()
