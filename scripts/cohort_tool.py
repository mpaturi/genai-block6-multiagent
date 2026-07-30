"""Role 2's Cypher queries and Neo4j driver (see docs/spec.md §2, docs/plan.md §6/§7).

FULL_COHORT_QUERY_TEMPLATE is Block 5's VERIFY_PATIENTS_QUERY_TEMPLATE
(genai-block5-agent/block5_agent/graph_tool.py) with just its trailing
"AND p.person_id IN $person_ids" clause dropped - the whole point of Role
2 is enumerating every matching patient, not just a candidate list handed
to it. condition and value are always bound as Cypher $parameters, never
interpolated into the query text - lab_property and op are substituted
via .format(), but only ever from the fixed whitelists below, never from
raw user input, same as Block 5's own precedent.

Read-only throughout: MATCH/RETURN only, no CREATE/MERGE/DELETE/SET.
"""
import functools
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Query

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# Matches Block 5's count_drugs precedent (graph_tool.py) - the starting
# value for the Cypher call's own timeout (docs/plan.md §7).
GRAPH_QUERY_TIMEOUT = 10

_LAB_PROPERTY = {
    "SBP": "latest_sbp",
    "BMI": "latest_bmi",
    "Glucose": "latest_glucose",
    "HbA1c": "latest_hba1c",
}
# Public alias so other modules (scripts/vocabulary_check.py) can see the
# full lab-name -> Patient-property mapping this repo's Cypher uses,
# without reaching into the private _LAB_PROPERTY mapping directly - lets
# vocabulary_check.py verify each mapped property name still actually
# exists on Patient nodes in Block 3's graph, not just trust this repo's
# own list of lab display names.
LAB_PROPERTY_NAMES = dict(_LAB_PROPERTY)
_COMPARISON_OP = {"above": ">", "below": "<"}

# The unbounded enumeration query - condition/value are Cypher
# $parameters, never string-interpolated. lab_property/op are filled in
# via .format() from the fixed whitelists above only.
FULL_COHORT_QUERY_TEMPLATE = """
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition {{condition_name: $condition}})
WHERE p.{lab_property} IS NOT NULL AND p.{lab_property} {op} $value
RETURN collect(DISTINCT p.person_id) AS matched_ids
"""

# The exhaustive drug-count query, over the full matched cohort - no
# top_k ceiling. A drug with zero matching patients simply has no row
# here, same as Block 5's DRUG_COUNT_QUERY.
EXHAUSTIVE_DRUG_COUNT_QUERY = """
MATCH (p:Patient)-[:PRESCRIBED]->(d:Drug)
WHERE p.person_id IN $person_ids
RETURN d.drug_name AS drug, count(DISTINCT p) AS patient_count
"""


class CohortServiceError(Exception):
    """Raised on any Cohort Agent tool failure - mirrors Block 5's
    GraphServiceError/RAGServiceError shape (detail, retryable) so
    run_cohort_agent's retry loop can check `retryable` the same way
    Block 5's run_agent already does.
    """

    def __init__(self, detail: str, retryable: bool = True):
        super().__init__(detail)
        self.detail = detail
        self.retryable = retryable


@functools.lru_cache(maxsize=1)
def get_driver():
    """One long-lived, reused Neo4j driver instance (docs/plan.md §6) -
    cached so a fresh driver is never constructed per query. Injectable
    via the `driver` keyword argument on the functions below, so tests
    can swap in a fake driver without touching this factory at all.
    """
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _validate_patient_ids(patient_ids: list) -> None:
    for patient_id in patient_ids:
        is_whole_number = isinstance(patient_id, int) and not isinstance(patient_id, bool)
        if not is_whole_number or patient_id <= 0:
            raise CohortServiceError("invalid_patient_id", retryable=False)


def query_full_cohort(
    condition: str,
    lab: str,
    comparison: str,
    value: float,
    *,
    driver=None,
) -> dict:
    """Enumerate every patient matching condition/lab/comparison/value -
    no top_k ceiling. Returns {"patient_ids": [...]}.
    """
    lab_property = _LAB_PROPERTY.get(lab)
    op = _COMPARISON_OP.get(comparison)
    if lab_property is None or op is None:
        raise CohortServiceError("invalid_lab_or_comparison", retryable=False)

    try:
        driver = driver if driver is not None else get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            query_text = FULL_COHORT_QUERY_TEMPLATE.format(lab_property=lab_property, op=op)
            row = session.run(
                Query(query_text, timeout=GRAPH_QUERY_TIMEOUT),
                condition=condition,
                value=value,
            ).single()
            matched_ids = list(row["matched_ids"])
    except Exception as exc:
        raise CohortServiceError(type(exc).__name__)

    return {"patient_ids": matched_ids}


def count_drugs_exhaustive(
    patient_ids: list[int],
    drug_a: str,
    drug_b: str,
    *,
    driver=None,
) -> dict:
    """Count drug_a/drug_b over the full matched cohort - no top_k
    ceiling. Returns {"drug_a_count": int, "drug_b_count": int}.
    """
    if not patient_ids:
        # Nothing to count - return immediately without opening a session.
        return {"drug_a_count": 0, "drug_b_count": 0}

    # Validated before any database interaction, per this project's
    # input-validation convention - fail fast locally rather than
    # sending bad data to the graph.
    _validate_patient_ids(patient_ids)

    try:
        driver = driver if driver is not None else get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                Query(EXHAUSTIVE_DRUG_COUNT_QUERY, timeout=GRAPH_QUERY_TIMEOUT),
                person_ids=patient_ids,
            )
            drug_counts = {row["drug"]: row["patient_count"] for row in rows}
    except Exception as exc:
        raise CohortServiceError(type(exc).__name__)

    return {
        "drug_a_count": drug_counts.get(drug_a, 0),
        "drug_b_count": drug_counts.get(drug_b, 0),
    }
