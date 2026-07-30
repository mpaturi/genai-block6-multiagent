"""run_cohort_agent - Role 2's entry point (see docs/spec.md §2, docs/plan.md §1/§7).

A plain retry loop, not an internal LangGraph: Role 2 is one combined
enumerate-and-count Cypher call, not a multi-step pipeline with a
sub-step worth modeling as its own graph. Never raises on tool failure -
same "never raise" contract as Block 5's run_agent - it degrades to a
CohortResult with outcome="tool_error" instead.
"""
import time

from block5_agent.schemas import QuestionInput

from scripts.cohort_tool import CohortServiceError, count_drugs_exhaustive, query_full_cohort
from scripts.schemas import CohortResult

# 2 retries => 3 attempts total, matching Block 5's agent.py:33
# _MAX_TOOL_RETRIES convention (docs/spec.md §2).
_MAX_TOOL_RETRIES = 2

# Backoff between retry attempts, in seconds - kept small (0.5s, 1.0s for
# the two possible retries) so a real Neo4j hiccup gets a moment to clear
# before trying again, without meaningfully slowing down a caller waiting
# on the result. Note: Block 5's own agent.py has the identical
# zero-delay-between-retries gap and was not touched here - that's a
# separate, optional follow-up for Block 5's own repo, not something this
# repo can fix by proxy.
_RETRY_BACKOFF_SECONDS = 0.5


def _call_with_retries(fn, *args, sleep_fn=time.sleep):
    """Call fn(*args), retrying up to _MAX_TOOL_RETRIES times on a
    retryable CohortServiceError, with a short backoff between attempts.
    Returns (result, None) on success, or (None, last_error_detail) once
    retries are exhausted or a non-retryable error is hit (bad input
    never gets retried, since trying again won't fix it).

    Catches any exception, not just CohortServiceError - anything else
    (a bug outside fn's own documented contract) is converted into one
    first, same as scripts/cohort_tool.py's own catch-all conversion, so
    this "never raises" contract holds on its own, not only when the
    orchestrator's outer node wrapper happens to also be there to catch it.

    sleep_fn is injectable (defaults to the real time.sleep) so tests can
    pass a no-op fake instead of actually waiting out the backoff.
    """
    last_error_detail = None
    for attempt in range(_MAX_TOOL_RETRIES + 1):
        try:
            return fn(*args), None
        except Exception as exc:
            service_error = exc if isinstance(exc, CohortServiceError) else CohortServiceError(
                type(exc).__name__
            )
            last_error_detail = service_error.detail
            if not service_error.retryable:
                return None, last_error_detail
            is_last_attempt = attempt == _MAX_TOOL_RETRIES
            if not is_last_attempt:
                sleep_fn(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    return None, last_error_detail


def run_cohort_agent(
    question: QuestionInput,
    *,
    graph_query_fn=query_full_cohort,
    count_fn=count_drugs_exhaustive,
    sleep_fn=time.sleep,
) -> CohortResult:
    """Enumerate every patient matching the question's structured filter,
    then count drug_a/drug_b over the full matched cohort - no top_k
    ceiling anywhere in this path.

    sleep_fn is injectable (see _call_with_retries) so tests can skip the
    real retry backoff delay.
    """
    # Step 1: enumerate the full cohort. If this fails after retries,
    # there is nothing to count - degrade straight to tool_error.
    query_result, error_detail = _call_with_retries(
        graph_query_fn,
        question.condition,
        question.lab,
        question.comparison,
        question.value,
        sleep_fn=sleep_fn,
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
        count_fn, patient_ids, question.drug_a, question.drug_b, sleep_fn=sleep_fn
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
