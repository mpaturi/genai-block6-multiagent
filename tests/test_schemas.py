"""Tests for scripts/schemas.py (see docs/spec.md §3).

These only exercise the schema definitions themselves - no fakes needed,
since there's no behavior here yet, just field shapes. The one thing
worth guarding explicitly: CohortResult.question stays QuestionInput
(structured - Role 2 never touches RAG) while MultiAgentAnswer.question
stays str (stringified, matching ClinicalAnswer's existing convention).
These are easy to accidentally let drift to the same type since both are
just called "question" - the tests below pin each to its own type so a
future edit that quietly changes one can't slip by unnoticed.
"""
from typing import Optional, get_type_hints

import pytest
from block5_agent.schemas import ClinicalAnswer, QuestionInput
from pydantic import ValidationError

from scripts.schemas import (
    Citation,
    CohortResult,
    MultiAgentAnswer,
    MultiAgentState,
    ReconciliationResult,
)

QUESTION = QuestionInput(
    condition="hypertension",
    lab="SBP",
    comparison="above",
    value=140,
    drug_a="Lisinopril",
    drug_b="Amlodipine",
)


def _cohort_result(**overrides):
    fields = {
        "question": QUESTION,
        "total_patients_matched": 40,
        "drug_a_count": 10,
        "drug_b_count": 5,
        "patient_ids": [1, 2, 3],
        "outcome": "answered",
    }
    fields.update(overrides)
    return CohortResult(**fields)


def _multi_agent_answer(**overrides):
    fields = {
        "question": "Of patients with hypertension and SBP > 140, how many are on Lisinopril vs. Amlodipine?",
        "answer": "40 patients matched; 10 on Lisinopril, 5 on Amlodipine.",
        "total_patients": 40,
        "drug_a_count": 10,
        "drug_b_count": 5,
        "confidence": "high",
        "mode": "reconciled",
        "citations": [],
        "discrepancy_flag": False,
    }
    fields.update(overrides)
    return MultiAgentAnswer(**fields)


def test_cohort_result_question_field_is_question_input_not_str():
    assert CohortResult.model_fields["question"].annotation is QuestionInput


def test_multi_agent_answer_question_field_is_str_not_question_input():
    assert MultiAgentAnswer.model_fields["question"].annotation is str


def test_multi_agent_answer_rejects_a_question_input_for_its_str_question_field():
    # Guards against the two "question" fields drifting to the same type -
    # if MultiAgentAnswer.question ever accidentally became QuestionInput
    # (or Any), this would stop raising.
    with pytest.raises(ValidationError):
        _multi_agent_answer(question=QUESTION)


def test_cohort_result_holds_a_real_question_input_instance():
    result = _cohort_result()
    assert isinstance(result.question, QuestionInput)
    assert result.question == QUESTION


def test_cohort_result_drug_counts_are_not_required_to_sum_to_total():
    # A patient can be on neither drug, one, or both - this is not a
    # partition, so drug_a_count + drug_b_count > total is legal.
    result = _cohort_result(total_patients_matched=5, drug_a_count=4, drug_b_count=4)
    assert result.total_patients_matched == 5
    assert result.drug_a_count == 4
    assert result.drug_b_count == 4


def test_cohort_result_outcome_rejects_a_value_outside_its_literal():
    with pytest.raises(ValidationError):
        _cohort_result(outcome="succeeded")


def test_cohort_result_caveat_defaults_to_none():
    assert _cohort_result().caveat is None


def test_reconciliation_result_authoritative_source_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ReconciliationResult(
            counts_match=False,
            authoritative_source="both",
            discrepancy_flag=True,
            notes="disagreement",
        )


def test_reconciliation_result_accepts_neither_as_authoritative_source():
    result = ReconciliationResult(
        counts_match=False,
        authoritative_source="neither",
        discrepancy_flag=True,
        notes="counts disagree",
    )
    assert result.authoritative_source == "neither"
    assert result.discrepancy_flag is True


def test_citation_holds_patient_id_snippet_and_source():
    citation = Citation(patient_id=1, snippet="Patient 1 text.", source="clinical")
    assert citation.patient_id == 1
    assert citation.snippet == "Patient 1 text."
    assert citation.source == "clinical"


def test_multi_agent_answer_mode_rejects_a_value_outside_its_literal():
    with pytest.raises(ValidationError):
        _multi_agent_answer(mode="partially_reconciled")


def test_multi_agent_answer_confidence_rejects_a_value_outside_its_literal():
    with pytest.raises(ValidationError):
        _multi_agent_answer(confidence="very_high")


def test_multi_agent_answer_citations_is_a_list_of_citation_objects():
    citation = Citation(patient_id=1, snippet="Patient 1 text.", source="clinical")
    answer = _multi_agent_answer(citations=[citation])
    assert answer.citations == [citation]
    assert isinstance(answer.citations[0], Citation)


def test_multi_agent_answer_caveat_defaults_to_none():
    assert _multi_agent_answer().caveat is None


def test_multi_agent_state_question_field_is_question_input():
    hints = get_type_hints(MultiAgentState)
    assert hints["question"] is QuestionInput


def test_multi_agent_state_clinical_result_is_optional_clinical_answer():
    hints = get_type_hints(MultiAgentState)
    assert hints["clinical_result"] == Optional[ClinicalAnswer]


def test_multi_agent_state_cohort_result_is_optional_cohort_result():
    hints = get_type_hints(MultiAgentState)
    assert hints["cohort_result"] == Optional[CohortResult]


def test_multi_agent_state_final_answer_is_optional_multi_agent_answer():
    hints = get_type_hints(MultiAgentState)
    assert hints["final_answer"] == Optional[MultiAgentAnswer]


def test_multi_agent_state_reconciliation_is_optional_reconciliation_result():
    hints = get_type_hints(MultiAgentState)
    assert hints["reconciliation"] == Optional[ReconciliationResult]


def test_multi_agent_state_clinical_count_step_ran_is_the_carried_forward_bool():
    # Carries forward the second value of Block 5's
    # run_agent(...) -> tuple[ClinicalAnswer, bool] (see docs/spec.md §3) -
    # otherwise silently dropped by this repo's clinical branch node.
    hints = get_type_hints(MultiAgentState)
    assert hints["clinical_count_step_ran"] == Optional[bool]


@pytest.mark.parametrize("key", ["clinical_error_kind", "cohort_error_kind"])
def test_multi_agent_state_error_kind_fields_share_the_four_way_literal(key):
    hints = get_type_hints(MultiAgentState)
    error_kind_type = hints[key]
    # Optional[Literal[...]] - unwrap to inspect the four allowed values.
    args = error_kind_type.__args__
    literal_args = next(arg for arg in args if arg is not type(None))
    assert set(literal_args.__args__) == {
        "timeout",
        "connection_error",
        "validation_error",
        "unknown",
    }
