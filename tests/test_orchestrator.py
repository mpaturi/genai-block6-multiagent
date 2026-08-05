"""Tests for scripts/orchestrator.py (see docs/spec.md §2/§4, docs/plan.md
§4/§5/§8/§12/§15).

TDD: written before scripts/orchestrator.py exists - every test here fails
with a ModuleNotFoundError until Phase 3. clinical_agent_fn/cohort_agent_fn
are faked throughout (spec.md §2's DI pattern), so no real run_agent,
run_cohort_agent, LLM, RAG, or Neo4j call happens anywhere in this file.

Contract decisions this phase settles, where spec.md/plan.md leave the
exact mechanics implicit (Phase 3 must satisfy these, not a different
reading):

- The degradation matrix's "tool_error" rows are driven by
  ClinicalAnswer.outcome/CohortResult.outcome == "tool_error" - a normal,
  non-raising return, per Block 5's own "never raises" contract for
  run_agent (spec.md §2) and this repo's identical contract for
  run_cohort_agent. state["clinical_error"]/state["cohort_error"] (and
  the _error_kind fields) are reserved for the node wrapper's own second
  line of defense (plan.md §5) - an exception the agent function raised
  outside its documented contract, or a real asyncio.TimeoutError. Both
  "documented tool_error" and "undocumented exception" must fold into the
  same degraded-mode bucket in reconcile_node - see
  test_out_of_contract_exception_is_caught_by_node_wrapper below.
- Citations in the final answer carry source="clinical" - Role 2 never
  produces citations, so every Citation in a reconciled or
  clinical_only_degraded answer traces back to Role 1's rag_citations.
- ReconciliationResult.notes (spec.md §3) has no field of its own on
  MultiAgentAnswer - the closest fit is MultiAgentAnswer.caveat, so this
  phase settles that reconcile_node's notes get surfaced through caveat,
  the one text field MultiAgentAnswer exposes for exactly this kind of
  human-review context.
- get_known_vocabulary (plan.md §12) is imported into scripts.orchestrator
  by name, so tests monkeypatch scripts.orchestrator.get_known_vocabulary
  directly rather than adding a keyword-only override to run_multi_agent's
  public signature, which spec.md §2 fixes as exactly
  (question, *, clinical_agent_fn=run_agent, cohort_agent_fn=run_cohort_agent).
  It returns {"conditions": set[str], "labs": set[str]}.
- both_failed's fixed answer text is "Both the clinical evidence agent and
  the cohort enumeration agent were unable to answer this question." -
  settled here since spec.md §4 only requires "explicit unable to answer
  text" without pinning exact wording.
"""
import asyncio
import logging
import time

from block5_agent.schemas import ClinicalAnswer

from scripts import orchestrator
from scripts.orchestrator import run_multi_agent, run_multi_agent_async
from scripts.schemas import CohortResult, Citation, MultiAgentAnswer, ReconciliationResult
from block5_agent.schemas import QuestionInput

QUESTION = QuestionInput(
    condition="hypertension",
    lab="SBP",
    comparison="above",
    value=140,
    drug_a="Lisinopril",
    drug_b="Amlodipine",
)

# run_agent's third return value (see block5_agent's phase-11-expose-cost
# branch) - fixed dummy values are fine here, these are fakes.
_DUMMY_COST_INFO = {"cost_usd": 0.001, "input_tokens": 50, "output_tokens": 20}


def _clinical_answer(
    rag_patient_ids,
    drug_counts,
    citations=None,
    outcome="answered",
    caveat=None,
) -> ClinicalAnswer:
    return ClinicalAnswer(
        question="Of patients with hypertension and SBP > 140, how many are on Lisinopril vs. Amlodipine?",
        answer="some patients matched",
        rag_patient_ids=rag_patient_ids,
        rag_citations=citations or [],
        graph_result=drug_counts,
        confidence="high",
        caveat=caveat,
        outcome=outcome,
    )


def _cohort_result(total, drug_a_count, drug_b_count, patient_ids=None, outcome="answered", caveat=None):
    return CohortResult(
        question=QUESTION,
        total_patients_matched=total,
        drug_a_count=drug_a_count,
        drug_b_count=drug_b_count,
        patient_ids=patient_ids if patient_ids is not None else list(range(1, total + 1)),
        outcome=outcome,
        caveat=caveat,
    )


def _fn(value):
    return lambda question: value


def _raising_fn(exc):
    def _fn(question):
        raise exc

    return _fn


def _run(clinical_fn, cohort_fn):
    return asyncio.run(
        run_multi_agent_async(QUESTION, clinical_agent_fn=clinical_fn, cohort_agent_fn=cohort_fn)
    )


# --- §2 reconciliation rules -------------------------------------------


def test_both_answered_matching_counts_at_or_under_25_is_high_confidence_reconciled():
    citations = [{"patient_id": 1, "chunk_id": "1_chunk0", "snippet": "Patient 1 text."}]
    clinical_fn = _fn(
        (
            _clinical_answer([1, 2, 3], {"Lisinopril": 2, "Amlodipine": 1}, citations=citations),
            True,
            _DUMMY_COST_INFO,
        )
    )
    cohort_fn = _fn(_cohort_result(3, 2, 1))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "reconciled"
    assert result.confidence == "high"
    assert result.total_patients == 3
    assert result.drug_a_count == 2
    assert result.drug_b_count == 1
    assert result.discrepancy_flag is False
    assert result.citations == [Citation(patient_id=1, snippet="Patient 1 text.", source="clinical")]


def test_both_answered_over_25_uses_cohorts_exhaustive_counts_as_authoritative():
    # Role 1's own count is capped at 25 by construction - its numbers here
    # (16/9 over 25 patients) disagree with Role 2's exhaustive 25/15 over
    # 40, but that's expected, not a real discrepancy: Role 2 is used
    # as-is, with no discrepancy_flag raised over it.
    clinical_fn = _fn(
        (_clinical_answer(list(range(1, 26)), {"Lisinopril": 16, "Amlodipine": 9}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(40, 25, 15))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "reconciled"
    assert result.confidence == "high"
    assert result.total_patients == 40
    assert result.drug_a_count == 25
    assert result.drug_b_count == 15
    assert result.discrepancy_flag is False
    assert result.caveat is not None
    assert "25" in result.caveat  # notes the citation-coverage gap


def test_both_answered_counts_disagree_at_or_under_25_is_low_confidence_discrepancy():
    clinical_fn = _fn(
        (_clinical_answer([1, 2, 3], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(3, 2, 1))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "reconciled"
    assert result.confidence == "low"
    assert result.discrepancy_flag is True
    assert result.caveat is not None
    # Both numbers surfaced for human review, per spec.md §2.
    assert "1" in result.caveat and "2" in result.caveat


def test_both_nothing_found_is_high_confidence_reconciled_with_zero_patients():
    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="nothing_found"))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "reconciled"
    assert result.confidence == "high"
    assert result.total_patients == 0
    assert result.drug_a_count == 0
    assert result.drug_b_count == 0
    assert result.discrepancy_flag is False


def test_nothing_found_answered_split_reports_confirmed_vocabulary_mismatch(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_known_vocabulary",
        lambda: {"conditions": {"Essential hypertension"}, "labs": {"SBP"}},
    )
    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(12, 8, 4))

    result = _run(clinical_fn, cohort_fn)

    assert result.confidence == "low"
    assert result.discrepancy_flag is True
    # "hypertension" (QUESTION.condition) isn't in the known-vocabulary set
    # faked above ("Essential hypertension") - the real finding, not a
    # generic guess string.
    assert "not present" in result.caveat.lower()


def test_nothing_found_answered_split_reports_vocabulary_looks_consistent(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_known_vocabulary",
        lambda: {"conditions": {"hypertension"}, "labs": {"SBP"}},
    )
    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(12, 8, 4))

    result = _run(clinical_fn, cohort_fn)

    assert result.confidence == "low"
    assert result.discrepancy_flag is True
    assert "unexplained" in result.caveat.lower() or "consistent" in result.caveat.lower()


def test_answered_nothing_found_split_reports_confirmed_vocabulary_mismatch(monkeypatch):
    # The mirror of test_nothing_found_answered_split_reports_confirmed_
    # vocabulary_mismatch above: clinical answered (its own drug counts
    # happen to be 0/0) while cohort's exhaustive enumeration found
    # nothing at all. Without routing this to the same vocabulary-split
    # handling, 0==0 numerically "matches" and this genuine asymmetric
    # split would silently fall through to "both agree" instead.
    monkeypatch.setattr(
        orchestrator,
        "get_known_vocabulary",
        lambda: {"conditions": {"Essential hypertension"}, "labs": {"SBP"}},
    )
    clinical_fn = _fn((_clinical_answer([1, 2, 3], {}), True, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="nothing_found"))

    result = _run(clinical_fn, cohort_fn)

    assert result.discrepancy_flag is True
    assert result.confidence == "low"
    assert "not present" in result.caveat.lower()


def test_answered_nothing_found_split_reports_vocabulary_looks_consistent(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_known_vocabulary",
        lambda: {"conditions": {"hypertension"}, "labs": {"SBP"}},
    )
    clinical_fn = _fn((_clinical_answer([1, 2, 3], {}), True, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="nothing_found"))

    result = _run(clinical_fn, cohort_fn)

    assert result.discrepancy_flag is True
    assert result.confidence == "low"
    assert "unexplained" in result.caveat.lower() or "consistent" in result.caveat.lower()


def test_clinical_count_step_ran_false_is_not_treated_as_a_zero_count(monkeypatch):
    # count_step_ran=False (Role 1 short-circuited before counting) must
    # route to the same "no comparable count" handling as the
    # nothing_found/answered split above, never as a literal 0 that
    # happens to be smaller than cohort's small-but-real count. This
    # routes through the same vocabulary-check path as those tests, so it
    # needs the same fake - without it, this test was making a real,
    # live Neo4j call (a Phase 2 bug: caught when Phase 3's orchestrator
    # actually exercised this path).
    monkeypatch.setattr(
        orchestrator, "get_known_vocabulary", lambda: {"conditions": {"hypertension"}, "labs": {"SBP"}}
    )
    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    result = _run(clinical_fn, cohort_fn)

    assert result.discrepancy_flag is True
    assert result.total_patients == 2  # cohort's real count, not clinical's implied 0
    assert result.confidence == "low"


# --- clinical_cost_info threading (block5_agent's phase-11-expose-cost) -


def test_clinical_cost_info_reaches_state_correctly(monkeypatch):
    # Spies on the real reconcile_node by capturing the state dict it's
    # called with, then delegates to it - proves clinical_cost_info
    # actually lands in MultiAgentState by the time reconcile_node runs,
    # not just that clinical_node's return value looked right in isolation.
    captured_state = {}
    real_reconcile_node = orchestrator.reconcile_node

    def _spy_reconcile_node(state):
        captured_state.update(state)
        return real_reconcile_node(state)

    monkeypatch.setattr(orchestrator, "reconcile_node", _spy_reconcile_node)

    clinical_fn = _fn(
        (_clinical_answer([1, 2], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    _run(clinical_fn, cohort_fn)

    assert captured_state["clinical_cost_info"] == _DUMMY_COST_INFO


# --- §4 degradation matrix ----------------------------------------------


def test_clinical_tool_error_cohort_succeeds_is_cohort_only_degraded_high_confidence():
    clinical_fn = _fn(
        (_clinical_answer([], {}, outcome="tool_error", caveat="search failed"), False, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(18, 11, 7))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "cohort_only_degraded"
    # Role 2's exhaustive count is the sole source here, and it's
    # exhaustive by construction regardless of Role 1's availability -
    # must not fall through to medium/low just because Role 1 produced
    # nothing to check the count against (plan.md §8).
    assert result.confidence == "high"
    assert result.citations == []
    assert result.total_patients == 18
    assert result.drug_a_count == 11
    assert result.drug_b_count == 7
    # plan.md §15's fixed template, verbatim.
    assert result.answer == (
        f"Of {18} patients with {QUESTION.condition} and {QUESTION.lab} {QUESTION.comparison} {QUESTION.value}, "
        f"{11} are on {QUESTION.drug_a} and {7} are on {QUESTION.drug_b}. "
        "No supporting evidence citations are available for this run because the clinical evidence agent failed."
    )


def test_clinical_succeeds_cohort_tool_error_is_clinical_only_degraded_medium_at_or_above_15():
    citations = [{"patient_id": i, "chunk_id": f"{i}_chunk0", "snippet": f"Patient {i} text."} for i in range(1, 16)]
    clinical_fn = _fn(
        (
            _clinical_answer(
                list(range(1, 16)),
                {"Lisinopril": 9, "Amlodipine": 6},
                citations=citations,
                caveat="Only 15 matching patient(s) were checked.",
            ),
            True,
            _DUMMY_COST_INFO,
        )
    )
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="tool_error", caveat="graph down"))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "clinical_only_degraded"
    assert result.confidence == "medium"
    assert result.total_patients == 15
    assert result.drug_a_count == 9
    assert result.drug_b_count == 6
    assert len(result.citations) == 15
    assert result.citations[0].source == "clinical"


def test_clinical_succeeds_cohort_tool_error_is_clinical_only_degraded_low_below_15():
    clinical_fn = _fn(
        (
            _clinical_answer(
                list(range(1, 15)),
                {"Lisinopril": 8, "Amlodipine": 6},
                caveat="Only 14 matching patient(s) were checked.",
            ),
            True,
            _DUMMY_COST_INFO,
        )
    )
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="tool_error", caveat="graph down"))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "clinical_only_degraded"
    assert result.confidence == "low"
    assert result.total_patients == 14


def test_both_tool_error_is_both_failed_low_confidence_and_never_raises():
    clinical_fn = _fn(
        (_clinical_answer([], {}, outcome="tool_error", caveat="search failed"), False, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(0, 0, 0, patient_ids=[], outcome="tool_error", caveat="graph down"))

    # If run_multi_agent_async ever let a failure here propagate, the call
    # below itself would fail the test with an uncaught exception rather
    # than a normal assertion failure.
    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "both_failed"
    assert result.confidence == "low"
    assert result.total_patients == 0
    assert result.citations == []
    assert result.answer == (
        "Both the clinical evidence agent and the cohort enumeration agent "
        "were unable to answer this question."
    )


# --- §5 node-wrapper defensive try/except -------------------------------


def test_out_of_contract_exception_is_caught_by_node_wrapper_and_degrades_gracefully():
    # A bug in a test fake (or a real one in production) - not one of
    # run_agent's/run_cohort_agent's own documented failure modes. The
    # node wrapper's own try/except (plan.md §5) must still catch this,
    # classify it via classify_exception, and degrade rather than crash.
    clinical_fn = _raising_fn(KeyError("unexpected bug"))
    cohort_fn = _fn(_cohort_result(9, 5, 3))

    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "cohort_only_degraded"
    assert result.confidence == "high"


def test_reconcile_node_wraps_a_real_vocabulary_check_failure_and_degrades_gracefully(monkeypatch):
    # Unlike the split tests above, this does NOT monkeypatch
    # get_known_vocabulary itself away - that would avoid exercising the
    # real failure this test exists to cover. Only get_driver is faked
    # (the lowest boundary that can't use a live Neo4j connection in this
    # test suite), so reconcile_node's own reconciliation logic,
    # _vocabulary_split_answer, and the real get_known_vocabulary/
    # _fetch_known_vocabulary all run for real and hit a real exception -
    # reconcile_node's wrapper (mirroring clinical_node/cohort_node's own
    # try/except, plan.md §5) must catch it and degrade rather than
    # letting the graph invocation crash.
    from scripts import vocabulary_check

    monkeypatch.setattr(vocabulary_check, "_cached_vocabulary", None)
    monkeypatch.setattr(vocabulary_check, "_cached_at", 0.0)

    def _raising_get_driver():
        raise RuntimeError("cannot connect to Neo4j")

    monkeypatch.setattr(vocabulary_check, "get_driver", _raising_get_driver)

    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(12, 8, 4))

    # If reconcile_node ever let this propagate, the call below itself
    # would fail with an uncaught exception rather than a normal
    # assertion failure.
    result = _run(clinical_fn, cohort_fn)

    assert result.mode == "both_failed"
    assert result.confidence == "low"


def test_malformed_clinical_write_is_caught_logged_and_marked_suspect(caplog):
    # clinical_agent_fn is compromised/buggy in a way run_agent's own
    # documented contract never produces on its own: its first tuple
    # element isn't a real ClinicalAnswer at all. Proves
    # scripts/state_validation.py is actually wired into the real
    # run_multi_agent_async path (see tests/test_state_validation.py for
    # the isolated unit tests on validate_state_update itself), not just
    # correct in isolation.
    clinical_fn = _fn(("not a ClinicalAnswer object", True, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(12, 8, 4))

    with caplog.at_level(logging.WARNING, logger="scripts.state_validation"):
        result = _run(clinical_fn, cohort_fn)

    # The malformed value must never have been trusted into reconciliation -
    # it's routed into the same degraded-mode bucket a real clinical_node
    # exception already takes, not silently used as-is.
    assert result.mode == "cohort_only_degraded"
    assert result.total_patients == 12
    assert any("clinical" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_reconcile_error_answer_itself_raising_still_returns_a_valid_answer(monkeypatch):
    # Forces the second, inner layer of defense (docs/tasks.md "Block 6 -
    # state validation"): even if _reconcile_error_answer - the fallback
    # for a reconcile_node failure - itself raises, run_multi_agent_async
    # must still return a real MultiAgentAnswer, never let the exception
    # escape. Reuses the same real vocabulary-check failure as the test
    # above to reach reconcile_node_safe's except block in the first
    # place, then breaks the fallback handler itself on top of that.
    from scripts import vocabulary_check

    monkeypatch.setattr(vocabulary_check, "_cached_vocabulary", None)
    monkeypatch.setattr(vocabulary_check, "_cached_at", 0.0)

    def _raising_get_driver():
        raise RuntimeError("cannot connect to Neo4j")

    monkeypatch.setattr(vocabulary_check, "get_driver", _raising_get_driver)

    def _raising_reconcile_error_answer(question, exc):
        raise RuntimeError("the fallback handler is broken too")

    monkeypatch.setattr(orchestrator, "_reconcile_error_answer", _raising_reconcile_error_answer)

    clinical_fn = _fn((_clinical_answer([], {}, outcome="nothing_found"), False, _DUMMY_COST_INFO))
    cohort_fn = _fn(_cohort_result(12, 8, 4))

    # If this exception ever escaped, the call below would fail with an
    # uncaught RuntimeError rather than a normal assertion failure.
    result = _run(clinical_fn, cohort_fn)

    assert isinstance(result, MultiAgentAnswer)
    assert result.mode == "both_failed"
    assert result.confidence == "low"
    # A fixed literal, no computed fields, per the task's own wording.
    assert result.total_patients == 0
    assert result.citations == []


def test_reconcile_node_hang_past_the_timeout_still_returns_a_valid_answer(monkeypatch):
    # The actual regression this fix targets (docs/tasks.md "Block 6 -
    # state validation" follow-up): before this fix, reconcile_node_safe
    # had no timeout of its own - a reconcile_node call that hung (e.g. a
    # wedged Neo4j that stopped enforcing its own Query timeout) would
    # hang run_multi_agent_async indefinitely, outside every ceiling this
    # repo has. A fake sleep stands in for that hang here - a real,
    # genuinely slow query is already covered elsewhere
    # (tests/test_vocabulary_check.py's real-timeout test); this test's
    # job is only to prove reconcile_node_safe's own wait_for wrapping
    # actually fires and escalates, not to re-prove Neo4j's timeout
    # mechanics.
    #
    # The fake sleeps far longer (3s) than the mocked timeout (0.2s) on
    # purpose, and the elapsed-time assertion below is a real, self-
    # verifying lower/upper bound - not just an observation made once by
    # hand. If reconcile_node_safe's wait_for wrapping were ever removed
    # in the future, this call would block for the full 3s and this
    # assertion would fail outright, not just make the test run slower.
    monkeypatch.setattr(orchestrator, "_RECONCILE_TIMEOUT_SECONDS", 0.2)

    def _hanging_reconcile_node(state):
        time.sleep(3)
        return {"reconciliation": None, "final_answer": None}

    monkeypatch.setattr(orchestrator, "reconcile_node", _hanging_reconcile_node)

    clinical_fn = _fn(
        (_clinical_answer([1, 2], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    started_at = time.monotonic()
    result = _run(clinical_fn, cohort_fn)
    elapsed = time.monotonic() - started_at

    assert isinstance(result, MultiAgentAnswer)
    assert result.mode == "both_failed"
    # Bounded by the mocked _RECONCILE_TIMEOUT_SECONDS (0.2s), nowhere
    # near the fake call's real 3s sleep.
    assert elapsed < 2.0


def test_final_answer_dropped_by_state_validation_is_escalated_to_the_fallback(monkeypatch):
    # The actual regression this fix targets (docs/tasks.md "Block 6 -
    # state validation" follow-up): validate_state_update silently drops
    # a malformed field when no error_key is given (reconcile_node's own
    # boundary) - correct for most fields, but final_answer isn't
    # optional the way the others are. Before this fix, a dropped
    # final_answer left reconcile_node_safe's returned dict without that
    # key at all, so state's initial None stayed in place all the way to
    # run_multi_agent_async's unguarded `final_answer.question` access -
    # an AttributeError, not a graceful degradation. This is the exact
    # scenario that must now crash *before* this fix and *not* crash
    # after it.
    def _fake_reconcile_node(state):
        return {
            "reconciliation": ReconciliationResult(
                counts_match=True, authoritative_source="cohort", discrepancy_flag=False, notes="ok"
            ),
            "final_answer": "not a real MultiAgentAnswer",
        }

    monkeypatch.setattr(orchestrator, "reconcile_node", _fake_reconcile_node)

    clinical_fn = _fn(
        (_clinical_answer([1, 2], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    # If the dropped final_answer were ever silently accepted, this call
    # would fail with an AttributeError (None has no attribute
    # 'question') rather than a normal assertion failure.
    result = _run(clinical_fn, cohort_fn)

    assert isinstance(result, MultiAgentAnswer)
    assert result.mode == "both_failed"


def test_valid_final_answer_with_another_malformed_field_does_not_over_escalate(monkeypatch):
    # Confirms the fix is scoped correctly: a malformed field *other than*
    # final_answer must still just be dropped by validate_state_update,
    # exactly as before this fix - not escalated into the reconciliation
    # fallback unnecessarily, since a perfectly valid final_answer is
    # still available to return as-is.
    real_final_answer = MultiAgentAnswer(
        question="Of patients with hypertension and SBP > 140, how many are on Lisinopril vs. Amlodipine?",
        answer="2 patients matched.",
        total_patients=2,
        drug_a_count=1,
        drug_b_count=1,
        confidence="high",
        mode="reconciled",
        citations=[],
        caveat=None,
        discrepancy_flag=False,
    )

    def _fake_reconcile_node(state):
        return {
            "reconciliation": "not a real ReconciliationResult",
            "final_answer": real_final_answer,
        }

    monkeypatch.setattr(orchestrator, "reconcile_node", _fake_reconcile_node)

    clinical_fn = _fn(
        (_clinical_answer([1, 2], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    result = _run(clinical_fn, cohort_fn)

    # The real final_answer is returned as-is - not replaced by
    # _reconcile_error_answer's generic both_failed fallback text/mode.
    assert result.answer == "2 patients matched."
    assert result.mode == "reconciled"
    assert result.total_patients == 2


def test_sync_entry_point_delegates_to_the_async_implementation():
    clinical_fn = _fn(
        (_clinical_answer([1, 2], {"Lisinopril": 1, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(2, 1, 1))

    result = run_multi_agent(QUESTION, clinical_agent_fn=clinical_fn, cohort_agent_fn=cohort_fn)

    assert result.mode == "reconciled"
    assert result.confidence == "high"


# --- §4/§7 branch timeout + thread-leak handling ------------------------


def _sleepy_clinical_fn(delay_seconds, result):
    def _fn(question):
        time.sleep(delay_seconds)
        return result

    return _fn


def test_branch_exceeding_the_ceiling_reports_timeout_via_the_dedicated_executor(monkeypatch):
    monkeypatch.setattr(orchestrator, "_BRANCH_TIMEOUT_SECONDS", 0.05)
    clinical_fn = _sleepy_clinical_fn(
        0.3, (_clinical_answer([1, 2, 3], {"Lisinopril": 2, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(9, 5, 3))

    result = _run(clinical_fn, cohort_fn)

    # Clinical never reported back in time - treated as a clinical
    # failure, same as an out-of-contract exception or a documented
    # tool_error, so this degrades to cohort_only_degraded.
    assert result.mode == "cohort_only_degraded"
    assert result.confidence == "high"


def test_block6_executor_is_a_dedicated_thread_pool_not_the_default_one():
    import concurrent.futures

    assert isinstance(orchestrator.block6_executor, concurrent.futures.ThreadPoolExecutor)


def test_late_arriving_result_after_timeout_is_logged_as_a_warning_not_dropped(monkeypatch, caplog):
    monkeypatch.setattr(orchestrator, "_BRANCH_TIMEOUT_SECONDS", 0.05)
    clinical_fn = _sleepy_clinical_fn(
        0.3, (_clinical_answer([1, 2, 3], {"Lisinopril": 2, "Amlodipine": 1}), True, _DUMMY_COST_INFO)
    )
    cohort_fn = _fn(_cohort_result(9, 5, 3))

    with caplog.at_level(logging.WARNING):
        _run(clinical_fn, cohort_fn)
        # The orphaned thread from the timed-out call above is still
        # running in block6_executor - give it time to actually finish
        # and fire its late-completion warning before checking caplog.
        time.sleep(0.5)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("clinical" in r.message.lower() for r in warnings)
    assert any(
        "timed out" in r.message.lower() or "after timeout" in r.message.lower()
        for r in warnings
    )
