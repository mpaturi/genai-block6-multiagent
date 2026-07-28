"""Shared types for Block 6 (see docs/spec.md §3).

QuestionInput and ClinicalAnswer are Block 5's own schemas, imported
as-is - Role 1 is reused unmodified, so this repo never redeclares its
types. Everything below is new to this repo: CohortResult (Role 2's
output), ReconciliationResult and MultiAgentAnswer (the orchestrator's
output), Citation (one cited piece of evidence inside MultiAgentAnswer),
and MultiAgentState (the LangGraph state every node reads and writes).
"""
from typing import Literal, Optional, TypedDict

from block5_agent.schemas import ClinicalAnswer, QuestionInput
from pydantic import BaseModel

Confidence = Literal["high", "medium", "low"]
CohortOutcome = Literal["answered", "nothing_found", "tool_error"]
AuthoritativeSource = Literal["clinical", "cohort", "neither"]
Mode = Literal["reconciled", "clinical_only_degraded", "cohort_only_degraded", "both_failed"]
ErrorKind = Literal["timeout", "connection_error", "validation_error", "unknown"]


class CohortResult(BaseModel):
    """Role 2's output (see docs/spec.md §3).

    question is kept structured (QuestionInput), not stringified - Role 2
    never touches RAG, so there's no reason to format it as text first.
    drug_a_count/drug_b_count are not required to sum to
    total_patients_matched: a patient can be on neither drug, one, or
    both, so this is not a partition and must never be treated as a
    sanity-check invariant.
    """

    question: QuestionInput
    total_patients_matched: int
    drug_a_count: int
    drug_b_count: int
    patient_ids: list[int]
    outcome: CohortOutcome
    caveat: Optional[str] = None


class ReconciliationResult(BaseModel):
    """The orchestrator's reconciliation verdict (see docs/spec.md §2/§3).

    authoritative_source="neither" covers both the nothing_found/answered
    split and any unresolved count mismatch - in neither case does one
    agent's number get silently picked over the other's.
    """

    counts_match: bool
    authoritative_source: AuthoritativeSource
    discrepancy_flag: bool
    notes: str


class Citation(BaseModel):
    """One cited piece of evidence backing MultiAgentAnswer.answer (see
    docs/spec.md §3). patient_id and snippet map directly from Role 1's
    ClinicalAnswer.rag_citations entries; source is this repo's own label
    for where a citation came from - not a field Block 5 provides.
    """

    patient_id: int
    snippet: str
    source: str


class MultiAgentAnswer(BaseModel):
    """The orchestrator's final structured answer (see docs/spec.md §3).

    question is stringified here (unlike CohortResult.question above),
    matching ClinicalAnswer.question's existing convention - this keeps
    the final answer object consistent with Block 5 rather than
    introducing a second convention.
    """

    question: str
    answer: str
    total_patients: int
    drug_a_count: int
    drug_b_count: int
    confidence: Confidence
    mode: Mode
    citations: list[Citation]
    caveat: Optional[str] = None
    discrepancy_flag: bool


class MultiAgentState(TypedDict):
    """The orchestrator's explicit, inspectable LangGraph state (see
    docs/spec.md §3). clinical_count_step_ran carries forward the second
    value of Block 5's run_agent(...) -> tuple[ClinicalAnswer, bool, dict] -
    otherwise silently dropped by this repo's clinical branch node. If
    False, Role 1 never reached its drug-counting step, so its counts are
    not a real second opinion to compare Role 2 against - treat this the
    same as Role 1 having no comparable count at all, not as a 0 count
    that happens to disagree with Role 2. clinical_cost_info carries
    forward run_agent's third value - {"cost_usd", "input_tokens",
    "output_tokens"} - for scripts/run_log.py (plan.md §9). Stays None
    on the cohort_only_degraded, both_failed, and out-of-contract-
    exception paths, since Role 1 never successfully returned on any of
    those.
    """

    question: QuestionInput
    clinical_result: Optional[ClinicalAnswer]
    clinical_count_step_ran: Optional[bool]
    clinical_cost_info: Optional[dict]
    clinical_error: Optional[str]
    clinical_error_kind: Optional[ErrorKind]
    cohort_result: Optional[CohortResult]
    cohort_error: Optional[str]
    cohort_error_kind: Optional[ErrorKind]
    reconciliation: Optional[ReconciliationResult]
    final_answer: Optional[MultiAgentAnswer]
