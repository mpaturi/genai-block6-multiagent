"""get_known_vocabulary - Block 3's real condition/lab vocabulary (see
docs/plan.md §12).

Two layers use this: the CI-time entry point below (Phase 4) that checks
the answerable fixed eval questions against it once, and reconcile_node (scripts/
orchestrator.py), which calls this at runtime for any question whenever
it hits a nothing_found/answered split - so a genuinely new, unfamiliar
condition or lab isn't mistaken for a graph-schema mismatch just because
it hasn't been added to a fixed list.

Cached with a TTL (docs/plan.md §12) rather than indefinitely, so a
condition/lab added to Block 3's graph after process startup - a real
concern once this becomes a long-running deployed service - doesn't get
silently misreported as unknown forever.
"""
import json
import sys
import time
from pathlib import Path

from neo4j import Query

from scripts.cohort_tool import GRAPH_QUERY_TIMEOUT, LAB_PROPERTY_NAMES, NEO4J_DATABASE, get_driver

# data/eval/questions.json, resolved relative to this file rather than the
# current working directory, so this script runs the same whether it's
# invoked as `python -m scripts.vocabulary_check` from the repo root or
# from anywhere else.
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"

# Re-query Block 3's graph if the cached vocabulary is older than this,
# rather than caching indefinitely (docs/plan.md §12).
_CACHE_TTL_SECONDS = 15 * 60

_cached_vocabulary = None
_cached_at = 0.0

_DISTINCT_CONDITION_NAMES_QUERY = """
MATCH (c:Condition)
RETURN DISTINCT c.condition_name AS condition_name
"""

# Labs are stored as Patient node properties (latest_sbp, latest_bmi, ...),
# not as their own nodes with a queryable "lab name" - so instead of a
# DISTINCT node-property query like the one above, this asks the graph
# which property keys actually exist on real Patient nodes.
_DISTINCT_PATIENT_PROPERTY_KEYS_QUERY = """
MATCH (p:Patient)
UNWIND keys(p) AS property_key
RETURN DISTINCT property_key
"""


def _fetch_known_vocabulary(*, driver=None) -> dict:
    """Query Block 3's graph for its real, current condition names and
    Patient-node property keys.

    Labs are cross-referenced against the real graph the same way
    conditions are, not just trusted from scripts/cohort_tool.py's own
    LAB_PROPERTY_NAMES mapping - a lab name only counts as known here if
    its mapped property (e.g. "SBP" -> latest_sbp) actually exists on a
    Patient node right now. Trusting the mapping alone would miss a
    real-world property rename in Block 3's graph entirely: the lab name
    would still look "known", but every query using it would then
    silently match zero patients, with no error anywhere to catch it.
    """
    driver = driver if driver is not None else get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        condition_rows = session.run(Query(_DISTINCT_CONDITION_NAMES_QUERY, timeout=GRAPH_QUERY_TIMEOUT))
        conditions = {row["condition_name"] for row in condition_rows}
        property_key_rows = session.run(
            Query(_DISTINCT_PATIENT_PROPERTY_KEYS_QUERY, timeout=GRAPH_QUERY_TIMEOUT)
        )
        real_property_keys = {row["property_key"] for row in property_key_rows}

    known_labs = {
        lab_name
        for lab_name, property_name in LAB_PROPERTY_NAMES.items()
        if property_name in real_property_keys
    }
    return {"conditions": conditions, "labs": known_labs}


def get_known_vocabulary(*, driver=None) -> dict:
    """Return {"conditions": set[str], "labs": set[str]} - Block 3's real,
    current vocabulary, cached for up to _CACHE_TTL_SECONDS.
    """
    global _cached_vocabulary, _cached_at
    now = time.monotonic()
    is_cache_fresh = _cached_vocabulary is not None and (now - _cached_at) < _CACHE_TTL_SECONDS
    if is_cache_fresh:
        return _cached_vocabulary

    _cached_vocabulary = _fetch_known_vocabulary(driver=driver)
    _cached_at = now
    return _cached_vocabulary


def _load_questions(questions_path: Path) -> list[dict]:
    """Load and validate data/eval/questions.json's shape - every question
    must have a condition and a lab string, since that's all this check
    cross-references. Fails loudly on a malformed file rather than
    letting a missing/blank field surface as a confusing KeyError deeper
    in the check below.
    """
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{questions_path} must contain a non-empty JSON array")

    for question in questions:
        question_id = question.get("id", "<missing id>")
        for field in ("condition", "lab"):
            value = question.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"question {question_id!r} is missing a valid '{field}' string")

    return questions


def check_vocabulary(*, driver=None, questions_path: Path = _QUESTIONS_PATH) -> list[str]:
    """Cross-reference every *answerable* question's condition/lab against
    Block 3's real, current vocabulary (docs/plan.md §12's CI-time layer).
    Returns a list of human-readable mismatch descriptions - empty if
    every answerable question's condition and lab has an exact match.

    Questions marked "answerable": false are skipped here on purpose -
    they exist specifically to use a condition/lab this system has never
    heard of (e.g. "schizophrenia"), so a vocabulary mismatch on one of
    those is the expected, correct outcome, not a drift bug to flag.
    Missing "answerable" is treated as true, so older question files
    without the field still get checked as before.
    """
    questions = _load_questions(questions_path)
    vocabulary = get_known_vocabulary(driver=driver)

    mismatches = []
    for question in questions:
        if not question.get("answerable", True):
            continue
        question_id = question["id"]
        condition = question["condition"]
        lab = question["lab"]
        if condition not in vocabulary["conditions"]:
            mismatches.append(
                f"question {question_id!r}: condition {condition!r} not found in Block 3's graph"
            )
        if lab not in vocabulary["labs"]:
            mismatches.append(f"question {question_id!r}: lab {lab!r} not a recognized lab name")

    return mismatches


def main() -> int:
    """CI-time entry point: exit 0 if every fixed eval question's
    condition/lab has an exact match in Block 3's real graph vocabulary,
    exit 1 and print every mismatch found otherwise (docs/plan.md §12).
    """
    mismatches = check_vocabulary()
    if mismatches:
        print("Vocabulary check FAILED:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return 1

    print("Vocabulary check passed: every fixed eval question's condition/lab matches Block 3's graph.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
