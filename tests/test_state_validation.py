"""Tests for scripts/state_validation.py (see genai-block7-security's
docs/spec.md LLM06 section: "Orchestrator state tampering... If a
compromised or simply buggy node wrote an unexpected value into it, the
reconciliation logic downstream would trust it without validation").
"""
import logging

from block5_agent.schemas import ClinicalAnswer

from scripts.schemas import CohortResult
from scripts.state_validation import validate_state_update

QUESTION_TEXT = "Of patients with hypertension and SBP > 140, how many are on Lisinopril vs. Amlodipine?"


def _real_clinical_answer() -> ClinicalAnswer:
    return ClinicalAnswer(
        question=QUESTION_TEXT,
        answer="some patients matched",
        rag_patient_ids=[1, 2, 3],
        rag_citations=[],
        graph_result={"Lisinopril": 2},
        confidence="high",
        caveat=None,
        outcome="answered",
    )


def _real_cohort_result() -> CohortResult:
    from block5_agent.schemas import QuestionInput

    question = QuestionInput(
        condition="hypertension", lab="SBP", comparison="above", value=140,
        drug_a="Lisinopril", drug_b="Amlodipine",
    )
    return CohortResult(
        question=question,
        total_patients_matched=3,
        drug_a_count=2,
        drug_b_count=1,
        patient_ids=[1, 2, 3],
        outcome="answered",
        caveat=None,
    )


# --- well-formed updates pass through unchanged --------------------------


def test_well_formed_clinical_update_passes_through_unchanged():
    update = {
        "clinical_result": _real_clinical_answer(),
        "clinical_count_step_ran": True,
        "clinical_cost_info": {"cost_usd": 0.01, "input_tokens": 10, "output_tokens": 5},
    }
    result = validate_state_update(
        "clinical", update, error_key="clinical_error", error_kind_key="clinical_error_kind"
    )
    assert result == update


def test_well_formed_cohort_update_passes_through_unchanged():
    update = {"cohort_result": _real_cohort_result()}
    result = validate_state_update(
        "cohort", update, error_key="cohort_error", error_kind_key="cohort_error_kind"
    )
    assert result == update


def test_none_values_are_always_valid_since_every_field_is_optional():
    update = {"clinical_result": None, "clinical_count_step_ran": None, "clinical_cost_info": None}
    result = validate_state_update(
        "clinical", update, error_key="clinical_error", error_kind_key="clinical_error_kind"
    )
    assert result == update


# --- malformed updates get caught, logged, and routed to the error path --


def test_wrong_type_for_clinical_result_is_caught_and_routed_to_error_key():
    update = {"clinical_result": "not a ClinicalAnswer object", "clinical_count_step_ran": True}
    result = validate_state_update(
        "clinical", update, error_key="clinical_error", error_kind_key="clinical_error_kind"
    )
    assert result["clinical_error"] is not None
    assert result["clinical_error_kind"] == "validation_error"
    # The malformed value must never survive into the returned update -
    # this is what "not silently propagating it into reconciliation" means.
    assert "clinical_result" not in result


def test_wrong_type_for_boolean_field_is_caught():
    update = {"clinical_result": _real_clinical_answer(), "clinical_count_step_ran": "yes"}
    result = validate_state_update(
        "clinical", update, error_key="clinical_error", error_kind_key="clinical_error_kind"
    )
    assert result["clinical_error"] is not None
    assert result["clinical_error_kind"] == "validation_error"


def test_unrecognized_key_is_caught():
    update = {"cohort_result": _real_cohort_result(), "totally_unexpected_field": 42}
    result = validate_state_update(
        "cohort", update, error_key="cohort_error", error_kind_key="cohort_error_kind"
    )
    assert result["cohort_error"] is not None
    assert result["cohort_error_kind"] == "validation_error"


def test_malformed_update_logs_a_warning_naming_the_node(caplog):
    update = {"cohort_result": 12345}
    with caplog.at_level(logging.WARNING, logger="scripts.state_validation"):
        validate_state_update(
            "cohort", update, error_key="cohort_error", error_kind_key="cohort_error_kind"
        )
    assert any("cohort" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# --- no fallback error key (reconcile_node's own boundary): drop, don't replace-with-error --


def test_without_a_fallback_error_key_malformed_fields_are_dropped_not_replaced():
    update = {"reconciliation": "not a real ReconciliationResult", "final_answer": None}
    result = validate_state_update("reconcile_node", update)
    assert "reconciliation" not in result
    assert result.get("final_answer") is None


def test_without_a_fallback_error_key_well_formed_fields_still_pass_through():
    from scripts.schemas import ReconciliationResult

    reconciliation = ReconciliationResult(
        counts_match=True, authoritative_source="cohort", discrepancy_flag=False, notes="ok"
    )
    update = {"reconciliation": reconciliation, "final_answer": None}
    result = validate_state_update("reconcile_node", update)
    assert result == update
