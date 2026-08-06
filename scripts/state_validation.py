"""Schema/type validation for scripts.orchestrator's MultiAgentState node
boundaries (see genai-block7-security's docs/tasks.md "Block 6 - state
validation" phase, docs/spec.md LLM06: "Orchestrator state tampering:
MultiAgentState is a mutable TypedDict passed between LangGraph nodes. If
a compromised or simply buggy node wrote an unexpected value into it, the
reconciliation logic downstream would trust it without validation").

MultiAgentState (scripts/schemas.py) is a plain TypedDict - LangGraph
merges whatever dict a node returns straight into shared state, with no
runtime check that a value actually matches its declared type.
validate_state_update is the one place that check happens, for every node
that writes to it.
"""
import logging

from block5_agent.schemas import ClinicalAnswer

from scripts.schemas import CohortResult, MultiAgentAnswer, ReconciliationResult

logger = logging.getLogger(__name__)

_VALID_ERROR_KINDS = {"timeout", "connection_error", "validation_error", "unknown"}

# Expected type for every field a node is allowed to write into
# MultiAgentState (scripts/schemas.py) - None is always additionally
# valid, since every one of these fields is Optional there. "question" is
# deliberately absent: it's set once by the initial state, never written
# by a node, so a node returning it at all is itself a boundary violation.
_FIELD_TYPES: dict[str, type] = {
    "clinical_result": ClinicalAnswer,
    "clinical_count_step_ran": bool,
    "clinical_cost_info": dict,
    "clinical_error": str,
    "clinical_error_kind": str,
    "cohort_result": CohortResult,
    "cohort_error": str,
    "cohort_error_kind": str,
    "reconciliation": ReconciliationResult,
    "final_answer": MultiAgentAnswer,
}

_ERROR_KIND_FIELDS = {"clinical_error_kind", "cohort_error_kind"}


def _find_violations(update: dict) -> dict[str, str]:
    """Returns {field_name: violation_message} for every key in update
    that fails validation - empty if update is entirely well-formed."""
    violations = {}
    for key, value in update.items():
        expected_type = _FIELD_TYPES.get(key)
        if expected_type is None:
            violations[key] = f"{key!r} is not a recognized MultiAgentState field"
            continue
        if value is not None and not isinstance(value, expected_type):
            violations[key] = (
                f"{key!r} expected {expected_type.__name__} or None, got {type(value).__name__}"
            )
            continue
        if key in _ERROR_KIND_FIELDS and value is not None and value not in _VALID_ERROR_KINDS:
            violations[key] = f"{key!r} has an unrecognized error kind: {value!r}"
    return violations


def validate_state_update(
    node_name: str,
    update: dict,
    *,
    error_key: str | None = None,
    error_kind_key: str | None = None,
) -> dict:
    """Checks every key/value pair a node is about to write into
    MultiAgentState against its declared type, before it ever reaches
    shared state.

    On success, returns update unchanged. On any violation, logs a
    warning naming the node and every offending field, then:
    - if error_key/error_kind_key are given (clinical_node/cohort_node's
      boundary), the malformed update is entirely replaced by that
      branch's existing error-state pair, marking the whole branch
      suspect and routing it into the degradation matrix's already-tested
      tool_error handling (docs/plan.md §5) - the same path a real
      exception already takes, so a compromised/buggy agent function
      can't get a partially-trusted result into reconciliation.
    - otherwise (reconcile_node's own boundary, which has no equivalent
      branch-level error state to fall back on - its own unguarded-helper
      failure mode is handled separately, see _reconcile_error_answer),
      just the offending fields are dropped and the rest of the update is
      kept, so a single stray key can't block otherwise-valid output.
    """
    violations = _find_violations(update)
    if not violations:
        return update

    logger.warning(
        "%s wrote a malformed MultiAgentState update, marking it suspect: %s",
        node_name,
        "; ".join(violations.values()),
    )

    if error_key is not None:
        return {error_key: "; ".join(violations.values()), error_kind_key: "validation_error"}

    return {key: value for key, value in update.items() if key not in violations}
