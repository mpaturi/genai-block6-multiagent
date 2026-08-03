"""run_cohort_agent - Role 2's entry point (see docs/spec.md §2, docs/plan.md §1/§7).

A plain retry loop, not an internal LangGraph: Role 2 is one combined
enumerate-and-count Cypher call, not a multi-step pipeline with a
sub-step worth modeling as its own graph. Never raises on tool failure -
same "never raise" contract as Block 5's run_agent - it degrades to a
CohortResult with outcome="tool_error" instead.
"""
from block5_agent.schemas import QuestionInput

from scripts.cohort_tool import CohortServiceError, count_drugs_exhaustive, query_full_cohort
from scripts.schemas import CohortResult

# 2 retries => 3 attempts total, matching Block 5's agent.py:33
# _MAX_TOOL_RETRIES convention (docs/spec.md §2).
_MAX_TOOL_RETRIES = 2


def _call_with_retries(fn, *args):
    """Call fn(*args), retrying up to _MAX_TOOL_RETRIES times on a
    retryable CohortServiceError. Returns (result, None) on success, or
    (None, last_error_detail) once retries are exhausted or a
    non-retryable error is hit (bad input never gets retried, since
    trying again won't fix it).

    Catches any exception, not just CohortServiceError - anything else
    (a bug outside fn's own documented contract) is converted into one
    first, same as scripts/cohort_tool.py's own catch-all conversion, so
    this "never raises" contract holds on its own, not only when the
    orchestrator's outer node wrapper happens to also be there to catch it.
    """
    last_error_detail = None
    for _attempt in range(_MAX_TOOL_RETRIES + 1):
        try:
            return fn(*args), None
        except Exception as exc:
            service_error = exc if isinstance(exc, CohortServiceError) else CohortServiceError(
                type(exc).__name__
            )
            last_error_detail = service_error.detail
            if not service_error.retryable:
                return None, last_error_detail
    return None, last_error_detail


def run_cohort_agent(
    question: QuestionInput,
    *,
    graph_query_fn=query_full_cohort,
    count_fn=count_drugs_exhaustive,
) -> CohortResult:
    """Enumerate every patient matching the question's structured filter,
    then count drug_a/drug_b over the full matched cohort - no top_k
    ceiling anywhere in this path.
    """
    # Step 1: enumerate the full cohort. If this fails after retries,
    # there is nothing to count - degrade straight to tool_error.
    query_result, error_detail = _call_with_retries(
        graph_query_fn, question.condition, question.lab, question.comparison, question.value
    )
    if query_result is None:
        return CohortResult(
            question=question,
            total_patients_matched=0,
            drug_a_count=0,
            drug_b_count=0,
            patient_ids=[],
            outcome="tool_error",
            caveat=(
                "The cohort enumeration query failed after repeated attempts "
                f"({error_detail})."
            ),
        )

    # Step 2: no matching patients - skip the count step entirely, same
    # as Block 5's own fallback-on-zero-matches behavior.
    patient_ids = query_result["patient_ids"]
    if not patient_ids:
        return CohortResult(
            question=question,
            total_patients_matched=0,
            drug_a_count=0,
            drug_b_count=0,
            patient_ids=[],
            outcome="nothing_found",
        )

    # Step 3: count drug_a/drug_b over the full matched cohort. If this
    # fails after retries, the patients were found but not counted -
    # still a tool_error, since CohortResult has no partial-success shape.
    count_result, error_detail = _call_with_retries(
        count_fn, patient_ids, question.drug_a, question.drug_b
    )
    if count_result is None:
        return CohortResult(
            question=question,
            total_patients_matched=0,
            drug_a_count=0,
            drug_b_count=0,
            patient_ids=[],
            outcome="tool_error",
            caveat=(
                "The exhaustive drug-count query failed after repeated attempts "
                f"({error_detail})."
            ),
        )

    return CohortResult(
        question=question,
        total_patients_matched=len(patient_ids),
        drug_a_count=count_result["drug_a_count"],
        drug_b_count=count_result["drug_b_count"],
        patient_ids=patient_ids,
        outcome="answered",
    )
