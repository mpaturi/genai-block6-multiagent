"""Tests for scripts/cohort_agent.py's run_cohort_agent (see docs/spec.md §2,
docs/plan.md §1/§7).

TDD: written before scripts/cohort_agent.py and scripts/cohort_tool.py
exist - these fail with an ImportError until Phase 3. graph_query_fn and
count_fn are faked throughout (docs/plan.md §2's DI pattern, mirroring
Block 5's search_fn/count_fn), so no real Neo4j connection is needed.

run_cohort_agent(question, *, graph_query_fn=query_full_cohort,
count_fn=count_drugs_exhaustive) -> CohortResult (plan.md §1) - a plain
retry loop, not an internal LangGraph (single combined enumerate-and-count
step, no multi-step pipeline to model). It never raises, even when a
fake raises on every call - same "never raise on tool failure" contract
as Block 5's run_agent (spec.md §2), degrading to outcome="tool_error"
instead.

graph_query_fn(condition, lab, comparison, value) -> {"patient_ids": [...]}
is this repo's own signature for the unbounded enumeration call - it takes
the question's structured filter fields directly (no candidate patient_ids
to verify against, unlike Block 5's count_drugs, since Role 2 has no RAG
step handing it a candidate list to begin with). count_fn(patient_ids,
drug_a, drug_b) -> {"drug_a_count": int, "drug_b_count": int} counts only
the two named drugs over the full matched cohort - CohortResult only ever
stores those two counts (spec.md §3), so there's no reason for count_fn to
return a full drug->count mapping the way Block 5's count_drugs does.
"""
from scripts.cohort_agent import _MAX_TOOL_RETRIES, run_cohort_agent
from scripts.cohort_tool import CohortServiceError
from block5_agent.schemas import QuestionInput

QUESTION = QuestionInput(
    condition="hypertension",
    lab="SBP",
    comparison="above",
    value=140,
    drug_a="Lisinopril",
    drug_b="Amlodipine",
)


class _CountingFake:
    """Wraps a function and records how many times it was called, and with
    what arguments - mirrors Block 5's tests/test_agent_answers.py fake so
    tests can assert both call count and forwarded arguments.
    """

    def __init__(self, fn):
        self._fn = fn
        self.call_count = 0
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


def _always_raise(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _never_called(name):
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return _fn


def test_answered_happy_path_returns_exhaustive_counts():
    graph_query_fn = _CountingFake(
        lambda condition, lab, comparison, value: {"patient_ids": [1, 2, 3, 4, 5]}
    )
    count_fn = _CountingFake(
        lambda patient_ids, drug_a, drug_b: {"drug_a_count": 3, "drug_b_count": 2}
    )

    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    assert graph_query_fn.call_count == 1
    call_args, _ = graph_query_fn.calls[0]
    assert call_args == (QUESTION.condition, QUESTION.lab, QUESTION.comparison, QUESTION.value)
    assert count_fn.call_count == 1
    count_call_args, _ = count_fn.calls[0]
    assert count_call_args == ([1, 2, 3, 4, 5], QUESTION.drug_a, QUESTION.drug_b)
    assert result.question == QUESTION
    assert result.total_patients_matched == 5
    assert result.drug_a_count == 3
    assert result.drug_b_count == 2
    assert result.patient_ids == [1, 2, 3, 4, 5]
    assert result.outcome == "answered"


def test_nothing_found_short_circuits_count_step():
    graph_query_fn = _CountingFake(
        lambda condition, lab, comparison, value: {"patient_ids": []}
    )
    count_fn = _CountingFake(_never_called("count_fn"))

    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    assert graph_query_fn.call_count == 1
    assert count_fn.call_count == 0
    assert result.total_patients_matched == 0
    assert result.drug_a_count == 0
    assert result.drug_b_count == 0
    assert result.patient_ids == []
    assert result.outcome == "nothing_found"


def test_graph_query_broken_after_retries_exhausted_returns_tool_error():
    graph_query_fn = _CountingFake(_always_raise(CohortServiceError("connection_error")))
    count_fn = _CountingFake(_never_called("count_fn"))

    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    # _MAX_TOOL_RETRIES retries => _MAX_TOOL_RETRIES + 1 attempts total,
    # matching Block 5's agent.py:33 retry convention (spec.md §2).
    assert graph_query_fn.call_count == _MAX_TOOL_RETRIES + 1
    assert count_fn.call_count == 0
    assert result.total_patients_matched == 0
    assert result.drug_a_count == 0
    assert result.drug_b_count == 0
    assert result.patient_ids == []
    assert result.outcome == "tool_error"
    assert result.caveat is not None


def test_count_step_broken_after_retries_exhausted_returns_tool_error():
    graph_query_fn = _CountingFake(
        lambda condition, lab, comparison, value: {"patient_ids": [1, 2, 3]}
    )
    count_fn = _CountingFake(_always_raise(CohortServiceError("ServiceUnavailable")))

    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    assert graph_query_fn.call_count == 1
    assert count_fn.call_count == _MAX_TOOL_RETRIES + 1
    assert result.total_patients_matched == 0
    assert result.drug_a_count == 0
    assert result.drug_b_count == 0
    assert result.patient_ids == []
    assert result.outcome == "tool_error"
    assert result.caveat is not None


def test_never_raises_when_graph_query_fn_raises_a_non_cohort_service_error():
    # A bug outside run_cohort_agent's own documented CohortServiceError
    # contract (e.g. a fake or a real tool raising something else
    # entirely) must still be caught and folded into the same retry/
    # tool_error handling - the "never raises" contract has to hold
    # standalone, not only when the orchestrator's outer node wrapper
    # happens to be there to catch it too.
    graph_query_fn = _CountingFake(_always_raise(RuntimeError("unexpected bug")))
    count_fn = _CountingFake(_never_called("count_fn"))

    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    assert graph_query_fn.call_count == _MAX_TOOL_RETRIES + 1
    assert count_fn.call_count == 0
    assert result.outcome == "tool_error"
    assert result.caveat is not None


def test_never_raises_even_on_a_non_retryable_error_on_every_call():
    # retryable=False (bad input, mirroring Block 5's invalid_person_id/
    # invalid_lab_or_comparison precedent) must fail fast without
    # exhausting the retry budget, and still degrade to a CohortResult
    # rather than propagating - distinct from the retry-exhaustion cases
    # above, which retry a *retryable* failure 3 times before giving up.
    graph_query_fn = _CountingFake(
        _always_raise(CohortServiceError("invalid_lab_or_comparison", retryable=False))
    )
    count_fn = _CountingFake(_never_called("count_fn"))

    # If run_cohort_agent ever let this propagate, pytest would fail this
    # test with an uncaught exception rather than a normal assertion
    # failure - the call below itself is the "never raises" assertion.
    result = run_cohort_agent(QUESTION, graph_query_fn=graph_query_fn, count_fn=count_fn)

    assert graph_query_fn.call_count == 1
    assert result.outcome == "tool_error"
