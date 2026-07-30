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

from scripts import vocabulary_check
from scripts.vocabulary_check import get_known_vocabulary


class _FakeSession:
    def __init__(self, condition_names, property_keys):
        self._condition_names = condition_names
        self._property_keys = property_keys

    def run(self, query):
        if "condition_name" in query:
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

    def session(self, database=None):
        return _FakeSession(self._condition_names, self._property_keys)


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
