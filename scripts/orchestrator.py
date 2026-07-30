"""The multi-agent orchestrator (see docs/spec.md §2/§4, docs/plan.md §3-8/§15).

MultiAgentState graph: dispatch fans out to clinical_node/cohort_node
(both async, run via the dedicated block6_executor thread pool so the two
synchronous agent calls actually run concurrently), which join at
reconcile_node.

Both public entry points are exposed per docs/plan.md §4: run_multi_agent
(sync, for this repo's own CLI/eval-harness use) delegates to
run_multi_agent_async (for callers already inside a running event loop,
e.g. a future FastAPI service).
"""
import asyncio
import concurrent.futures
import logging
import time

from block5_agent.agent import run_agent
from block5_agent.schemas import ClinicalAnswer, QuestionInput, assemble_question_text
from langgraph.graph import END, StateGraph

from scripts.cohort_agent import run_cohort_agent
from scripts.error_classification import classify_exception
from scripts.schemas import (
    Citation,
    CohortResult,
    MultiAgentAnswer,
    MultiAgentState,
    ReconciliationResult,
)
from scripts.vocabulary_check import get_known_vocabulary

logger = logging.getLogger(__name__)

# Best-effort supervisory ceiling per branch (docs/plan.md §7) - nearly 2x
# Role 1's ~80s theoretical worst case. Only catches hangs that occur
# before either tool's own internal timeout even starts its clock; each
# tool's own internal timeout is the real defense against a hung call.
_BRANCH_TIMEOUT_SECONDS = 150

# A dedicated thread pool (docs/plan.md §4), never the shared default pool
# asyncio.to_thread draws from - so a hung call here can never starve
# unrelated concurrent asyncio work elsewhere in the process.
block6_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="block6")

# Below this, Role 1's own patient count is still just its own best
# effort, not a confirmed exhaustive count (docs/plan.md §8).
_MEDIUM_CONFIDENCE_PATIENT_FLOOR = 15
# Above this, Role 1's count is known-incomplete by construction - Role
# 2's exhaustive count is used instead (docs/spec.md §2).
_ROLE1_TOP_K_CEILING = 25


async def _run_branch(branch_name: str, call_fn, on_success, error_keys: tuple[str, str]) -> dict:
    """Shared timeout + late-completion-warning wrapper for one branch
    (docs/plan.md §4/§5). call_fn takes no arguments and runs the
    synchronous agent call in the dedicated thread pool; on_success turns
    its raw return value into the state dict this node should return.
    """
    deadline = time.monotonic() + _BRANCH_TIMEOUT_SECONDS
    concurrent_future = block6_executor.submit(call_fn)

    def _warn_if_completed_after_the_branch_already_timed_out(fut: concurrent.futures.Future) -> None:
        # asyncio.wait_for's timeout below only stops *waiting* - it can't
        # kill the OS thread block6_executor started, so this callback is
        # the only place a late completion is ever noticed at all
        # (docs/plan.md §4's thread-leak note).
        if time.monotonic() > deadline:
            logger.warning(
                "%s branch timed out after %.0fs; its call actually completed "
                "after timeout - discarding the late result",
                branch_name,
                _BRANCH_TIMEOUT_SECONDS,
            )

    concurrent_future.add_done_callback(_warn_if_completed_after_the_branch_already_timed_out)

    error_key, error_kind_key = error_keys
    try:
        raw_result = await asyncio.wait_for(
            asyncio.wrap_future(concurrent_future), timeout=_BRANCH_TIMEOUT_SECONDS
        )
        return on_success(raw_result)
    except Exception as exc:
        # Second line of defense (docs/plan.md §5): run_agent/
        # run_cohort_agent are trusted to never raise on their own
        # documented failure modes - this catches anything outside that
        # contract (a bug, a real timeout) so the graph degrades instead
        # of crashing.
        return {error_key: str(exc), error_kind_key: classify_exception(exc)}


def _clinical_branch_failed(state: MultiAgentState) -> bool:
    """True for both an out-of-contract exception/timeout (clinical_error
    set) and a documented tool_error (a normal, non-raising return) -
    both must fold into the same degraded-mode bucket in reconcile_node.
    """
    if state.get("clinical_error") is not None:
        return True
    clinical_result = state.get("clinical_result")
    return clinical_result is not None and clinical_result.outcome == "tool_error"


def _cohort_branch_failed(state: MultiAgentState) -> bool:
    if state.get("cohort_error") is not None:
        return True
    cohort_result = state.get("cohort_result")
    return cohort_result is not None and cohort_result.outcome == "tool_error"


def _citations_from_clinical(clinical_result: ClinicalAnswer) -> list[Citation]:
    return [
        Citation(patient_id=entry["patient_id"], snippet=entry["snippet"], source="clinical")
        for entry in clinical_result.rag_citations
    ]


def _both_failed_answer(question: QuestionInput) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="neither",
        discrepancy_flag=False,
        notes="Both agents failed to produce an answer.",
    )
    final_answer = MultiAgentAnswer(
        question=assemble_question_text(question),
        answer=(
            "Both the clinical evidence agent and the cohort enumeration "
            "agent were unable to answer this question."
        ),
        total_patients=0,
        drug_a_count=0,
        drug_b_count=0,
        confidence="low",
        mode="both_failed",
        citations=[],
        caveat=None,
        discrepancy_flag=False,
    )
    return reconciliation, final_answer


def _cohort_only_degraded_answer(
    question: QuestionInput, cohort_result: CohortResult
) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    # Role 2's exhaustive count is the sole source here, and it's
    # exhaustive by construction regardless of Role 1's availability -
    # this is "high" confidence, never medium/low (docs/plan.md §8).
    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="cohort",
        discrepancy_flag=False,
        notes="Clinical evidence agent failed; cohort enumeration agent's exhaustive count used alone.",
    )
    # Fixed template (docs/plan.md §15) - deterministic, no LLM call, since
    # an LLM call here would reintroduce the exact nondeterminism/cost
    # Role 1's own failure was supposed to remove, and Role 1's LLM is
    # precisely what just failed in this mode.
    answer_text = (
        f"Of {cohort_result.total_patients_matched} patients with {question.condition} "
        f"and {question.lab} {question.comparison} {question.value}, "
        f"{cohort_result.drug_a_count} are on {question.drug_a} and "
        f"{cohort_result.drug_b_count} are on {question.drug_b}. "
        "No supporting evidence citations are available for this run because "
        "the clinical evidence agent failed."
    )
    final_answer = MultiAgentAnswer(
        question=assemble_question_text(question),
        answer=answer_text,
        total_patients=cohort_result.total_patients_matched,
        drug_a_count=cohort_result.drug_a_count,
        drug_b_count=cohort_result.drug_b_count,
        confidence="high",
        mode="cohort_only_degraded",
        citations=[],
        caveat=None,
        discrepancy_flag=False,
    )
    return reconciliation, final_answer


def _clinical_only_degraded_answer(
    question: QuestionInput, clinical_result: ClinicalAnswer
) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    patients_checked = len(clinical_result.rag_patient_ids)
    # Broadened medium band (docs/plan.md §8): without Role 2 confirming
    # it, even a Role-1 count that saturated its old top_k=25 ceiling is
    # still just Role 1's own best effort, not a confirmed exhaustive
    # count - so it belongs in medium, not high.
    confidence = "medium" if patients_checked >= _MEDIUM_CONFIDENCE_PATIENT_FLOOR else "low"
    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="clinical",
        discrepancy_flag=False,
        notes="Cohort enumeration agent failed; clinical evidence agent's answer used alone.",
    )
    final_answer = MultiAgentAnswer(
        question=clinical_result.question,
        answer=clinical_result.answer,
        total_patients=patients_checked,
        drug_a_count=clinical_result.graph_result.get(question.drug_a, 0),
        drug_b_count=clinical_result.graph_result.get(question.drug_b, 0),
        confidence=confidence,
        mode="clinical_only_degraded",
        citations=_citations_from_clinical(clinical_result),
        caveat=clinical_result.caveat,
        discrepancy_flag=False,
    )
    return reconciliation, final_answer


def _vocabulary_split_answer(
    question: QuestionInput, cohort_result: CohortResult
) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    # A nothing_found/answered split (or clinical_count_step_ran=False,
    # treated identically - docs/spec.md §3) is a genuine disagreement,
    # not a tool failure. Check the real vocabulary rather than writing a
    # generic guess into the notes (docs/plan.md §12).
    vocabulary = get_known_vocabulary()
    condition_known = question.condition in vocabulary["conditions"]
    lab_known = question.lab in vocabulary["labs"]
    if not condition_known or not lab_known:
        mismatched_field = "condition" if not condition_known else "lab"
        notes = f"confirmed: {mismatched_field} value not present in Block 3's graph"
    else:
        notes = "vocabulary looks consistent; disagreement is unexplained by this check"

    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="neither",
        discrepancy_flag=True,
        notes=notes,
    )
    final_answer = MultiAgentAnswer(
        question=assemble_question_text(question),
        answer=(
            "The clinical evidence agent and cohort enumeration agent disagree "
            "on whether any patients match this question - see caveat."
        ),
        total_patients=cohort_result.total_patients_matched,
        drug_a_count=cohort_result.drug_a_count,
        drug_b_count=cohort_result.drug_b_count,
        confidence="low",
        mode="reconciled",
        citations=[],
        caveat=notes,
        discrepancy_flag=True,
    )
    return reconciliation, final_answer


def _reconcile_error_answer(
    question: QuestionInput, exc: Exception
) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    """Fallback when reconcile_node itself raises - e.g. _vocabulary_split_
    answer's live get_known_vocabulary() Cypher call failing. Both branches
    may well have succeeded; it's reconciling their results that failed,
    but there's still no reliable answer to give, so this degrades into
    the same mode="both_failed" territory a clinical_node/cohort_node
    out-of-contract exception does (docs/plan.md §5), rather than letting
    the graph invocation crash.
    """
    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="neither",
        discrepancy_flag=False,
        notes=f"Reconciliation step failed ({classify_exception(exc)}): {exc}",
    )
    final_answer = MultiAgentAnswer(
        question=assemble_question_text(question),
        answer="The orchestrator was unable to reconcile the agents' results for this question.",
        total_patients=0,
        drug_a_count=0,
        drug_b_count=0,
        confidence="low",
        mode="both_failed",
        citations=[],
        caveat=None,
        discrepancy_flag=False,
    )
    return reconciliation, final_answer


def _both_nothing_found_answer(question: QuestionInput) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    reconciliation = ReconciliationResult(
        counts_match=True,
        authoritative_source="cohort",
        discrepancy_flag=False,
        notes="Both agents agree: no matching patients.",
    )
    final_answer = MultiAgentAnswer(
        question=assemble_question_text(question),
        answer="No patients match this question.",
        total_patients=0,
        drug_a_count=0,
        drug_b_count=0,
        confidence="high",
        mode="reconciled",
        citations=[],
        caveat=None,
        discrepancy_flag=False,
    )
    return reconciliation, final_answer


def _both_answered_reconciled_answer(
    question: QuestionInput, clinical_result: ClinicalAnswer, cohort_result: CohortResult
) -> tuple[ReconciliationResult, MultiAgentAnswer]:
    clinical_drug_a_count = clinical_result.graph_result.get(question.drug_a, 0)
    clinical_drug_b_count = clinical_result.graph_result.get(question.drug_b, 0)
    citations = _citations_from_clinical(clinical_result)

    if cohort_result.total_patients_matched > _ROLE1_TOP_K_CEILING:
        # Role 1's count is known-incomplete by definition once the true
        # cohort exceeds its top_k ceiling - Role 2's exhaustive count is
        # authoritative, and this disagreement is expected, not a real
        # discrepancy (docs/spec.md §2).
        reconciliation = ReconciliationResult(
            counts_match=False,
            authoritative_source="cohort",
            discrepancy_flag=False,
            notes=(
                f"Cohort enumeration agent's exhaustive count ({cohort_result.total_patients_matched} "
                f"patients) exceeds Role 1's {_ROLE1_TOP_K_CEILING}-patient citation ceiling."
            ),
        )
        caveat = (
            f"This cohort has {cohort_result.total_patients_matched} patients, more than the "
            f"{_ROLE1_TOP_K_CEILING} the clinical evidence agent can cite - citations below only "
            "cover the subset it retrieved, not the full cohort."
        )
        final_answer = MultiAgentAnswer(
            question=clinical_result.question,
            answer=clinical_result.answer,
            total_patients=cohort_result.total_patients_matched,
            drug_a_count=cohort_result.drug_a_count,
            drug_b_count=cohort_result.drug_b_count,
            confidence="high",
            mode="reconciled",
            citations=citations,
            caveat=caveat,
            discrepancy_flag=False,
        )
        return reconciliation, final_answer

    counts_match = (
        clinical_drug_a_count == cohort_result.drug_a_count
        and clinical_drug_b_count == cohort_result.drug_b_count
    )
    if counts_match:
        reconciliation = ReconciliationResult(
            counts_match=True,
            authoritative_source="cohort",
            discrepancy_flag=False,
            notes="Both agents agree.",
        )
        final_answer = MultiAgentAnswer(
            question=clinical_result.question,
            answer=clinical_result.answer,
            total_patients=cohort_result.total_patients_matched,
            drug_a_count=cohort_result.drug_a_count,
            drug_b_count=cohort_result.drug_b_count,
            confidence="high",
            mode="reconciled",
            citations=citations,
            caveat=None,
            discrepancy_flag=False,
        )
        return reconciliation, final_answer

    # Counts disagree despite both being within Role 1's own ceiling -
    # don't silently pick one, surface both for human review
    # (docs/spec.md §2).
    notes = (
        f"Clinical evidence agent found {clinical_drug_a_count}/{clinical_drug_b_count} "
        f"({question.drug_a}/{question.drug_b}); cohort enumeration agent found "
        f"{cohort_result.drug_a_count}/{cohort_result.drug_b_count} - disagreement not auto-resolved."
    )
    reconciliation = ReconciliationResult(
        counts_match=False,
        authoritative_source="neither",
        discrepancy_flag=True,
        notes=notes,
    )
    final_answer = MultiAgentAnswer(
        question=clinical_result.question,
        answer=clinical_result.answer,
        total_patients=cohort_result.total_patients_matched,
        drug_a_count=cohort_result.drug_a_count,
        drug_b_count=cohort_result.drug_b_count,
        confidence="low",
        mode="reconciled",
        citations=citations,
        caveat=notes,
        discrepancy_flag=True,
    )
    return reconciliation, final_answer


def reconcile_node(state: MultiAgentState) -> dict:
    """Implements every reconciliation rule from docs/spec.md §2 and the
    confidence redesign from docs/plan.md §8, once both branches have
    reported back.
    """
    question = state["question"]
    clinical_failed = _clinical_branch_failed(state)
    cohort_failed = _cohort_branch_failed(state)

    if clinical_failed and cohort_failed:
        reconciliation, final_answer = _both_failed_answer(question)
    elif clinical_failed:
        reconciliation, final_answer = _cohort_only_degraded_answer(question, state["cohort_result"])
    elif cohort_failed:
        reconciliation, final_answer = _clinical_only_degraded_answer(question, state["clinical_result"])
    else:
        clinical_result = state["clinical_result"]
        cohort_result = state["cohort_result"]
        both_nothing_found = (
            clinical_result.outcome == "nothing_found" and cohort_result.outcome == "nothing_found"
        )
        # clinical_count_step_ran=False means Role 1 never reached its
        # counting step, so it has no real second opinion to compare Role
        # 2 against - treat this the same as the nothing_found/answered
        # split below, never as a literal 0 (docs/spec.md §3).
        no_comparable_clinical_count = not state.get("clinical_count_step_ran")

        if both_nothing_found:
            reconciliation, final_answer = _both_nothing_found_answer(question)
        elif no_comparable_clinical_count and cohort_result.outcome == "answered":
            reconciliation, final_answer = _vocabulary_split_answer(question, cohort_result)
        elif clinical_result.outcome == "answered" and cohort_result.outcome == "nothing_found":
            # The mirror of the case just above: clinical answered while
            # cohort's exhaustive enumeration found nothing at all - a
            # genuine asymmetric split, not "both agree" just because both
            # sides' drug counts happen to be 0 (docs/spec.md §3).
            reconciliation, final_answer = _vocabulary_split_answer(question, cohort_result)
        else:
            reconciliation, final_answer = _both_answered_reconciled_answer(
                question, clinical_result, cohort_result
            )

    return {"reconciliation": reconciliation, "final_answer": final_answer}


async def run_multi_agent_async(
    question: QuestionInput,
    *,
    clinical_agent_fn=run_agent,
    cohort_agent_fn=run_cohort_agent,
) -> MultiAgentAnswer:
    """Run both agents concurrently via LangGraph, then reconcile. Never
    raises - every failure mode degrades to a fully-formed MultiAgentAnswer
    (docs/spec.md §4).
    """

    async def clinical_node(state: MultiAgentState) -> dict:
        def on_success(raw_result):
            answer, count_step_ran = raw_result
            return {"clinical_result": answer, "clinical_count_step_ran": count_step_ran}

        return await _run_branch(
            "clinical",
            lambda: clinical_agent_fn(state["question"]),
            on_success,
            ("clinical_error", "clinical_error_kind"),
        )

    async def cohort_node(state: MultiAgentState) -> dict:
        return await _run_branch(
            "cohort",
            lambda: cohort_agent_fn(state["question"]),
            lambda raw_result: {"cohort_result": raw_result},
            ("cohort_error", "cohort_error_kind"),
        )

    def dispatch(state: MultiAgentState) -> dict:
        return {}

    def reconcile_node_safe(state: MultiAgentState) -> dict:
        try:
            return reconcile_node(state)
        except Exception as exc:
            # Same second line of defense as clinical_node/cohort_node
            # (plan.md §5), extended to reconcile_node itself - this is
            # what actually catches _vocabulary_split_answer's live
            # get_known_vocabulary() Cypher call failing.
            reconciliation, final_answer = _reconcile_error_answer(state["question"], exc)
            return {"reconciliation": reconciliation, "final_answer": final_answer}

    # Built fresh per call (mirroring Block 5's run_agent) so each
    # invocation's clinical_agent_fn/cohort_agent_fn overrides are closed
    # over correctly - one dispatch node fans out to two branches, both
    # writing disjoint state keys, joined at reconcile_node.
    graph = StateGraph(MultiAgentState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("clinical", clinical_node)
    graph.add_node("cohort", cohort_node)
    graph.add_node("reconcile", reconcile_node_safe)
    graph.set_entry_point("dispatch")
    graph.add_edge("dispatch", "clinical")
    graph.add_edge("dispatch", "cohort")
    graph.add_edge("clinical", "reconcile")
    graph.add_edge("cohort", "reconcile")
    graph.add_edge("reconcile", END)
    compiled = graph.compile()

    initial_state: MultiAgentState = {
        "question": question,
        "clinical_result": None,
        "clinical_count_step_ran": None,
        "clinical_error": None,
        "clinical_error_kind": None,
        "cohort_result": None,
        "cohort_error": None,
        "cohort_error_kind": None,
        "reconciliation": None,
        "final_answer": None,
    }
    final_state = await compiled.ainvoke(initial_state)
    return final_state["final_answer"]


def run_multi_agent(
    question: QuestionInput,
    *,
    clinical_agent_fn=run_agent,
    cohort_agent_fn=run_cohort_agent,
) -> MultiAgentAnswer:
    """Sync entry point for this repo's own CLI/eval-harness usage -
    delegates to run_multi_agent_async so there's one real implementation,
    not two (docs/plan.md §4).
    """
    return asyncio.run(
        run_multi_agent_async(
            question, clinical_agent_fn=clinical_agent_fn, cohort_agent_fn=cohort_agent_fn
        )
    )
