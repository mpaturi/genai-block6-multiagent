"""Runs the full 12-question eval set through run_multi_agent_async and
scores it (see docs/spec.md §6, docs/plan.md §9/§13/§14).

The 9 answerable questions (q1-q8, q12) are scored for recall against
data/eval/answer_key.json's total_patients/drug_a_count/drug_b_count.
The 3 deliberately-unanswerable control questions (q9-q11) are scored
separately, pass/fail, on whether the system correctly reports zero
matching patients - never folded into the recall percentage. Any
discrepancy_flag=True on a scored question fails the build, naming
the question ID(s). A degradation-matrix dimension (fakes, no live calls)
asserts every row of spec.md §4's matrix still produces the correct
mode. A regression gate compares this run's recall and p95 latency
against data/eval/latency_baseline.json, writing that baseline file on
its first-ever real run.

Question/scored counts above are not hardcoded anywhere in the logic
below - every count derives from data/eval/questions.json's actual
contents (via "answerable", defaulting true) and answer_key.json, so
adding another question later doesn't require updating this file's own
assumptions about how many there are.
"""
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

from block5_agent.agent import run_agent
from block5_agent.schemas import ClinicalAnswer, QuestionInput

from scripts.orchestrator import run_multi_agent_async
from scripts.run_log import LOG_PATH
from scripts.schemas import CohortResult

QUESTIONS_PATH = Path("data/eval/questions.json")
ANSWER_KEY_PATH = Path("data/eval/answer_key.json")
FIXTURES_PATH = Path("data/eval/rag_fixtures.json")
BASELINE_PATH = Path("data/eval/latency_baseline.json")

# Matches Block 5's own CI gate precedent (see block5_agent/run_eval.py) -
# an absolute floor on top of the regression-vs-baseline check below.
RECALL_THRESHOLD = 0.70


# --- clinical_agent_fn construction (USE_RAG_FIXTURES / USE_STUB_ANSWER_FN) --


def _make_fixture_search_fn():
    """Mirrors Block 5's run_eval.py - a search_fn backed by recorded RAG
    fixtures, for CI, which has no live RAG service to call. Only the
    search step is faked; count_fn always stays real (the seeded local
    Neo4j container)."""
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    def _search_fn(
        query_text: str, condition=None, lab=None, comparison=None, value=None, top_k: int = 25
    ) -> dict:
        if query_text not in fixtures:
            raise RuntimeError(
                f"USE_RAG_FIXTURES is set, but no fixture is recorded for query "
                f"text {query_text!r}. Re-run block5_agent's capture_rag_fixtures.py "
                "if data/eval/questions.json changed."
            )
        return fixtures[query_text]

    return _search_fn


def _stub_answer_fn(question, rag_patient_ids, drug_a_count, drug_b_count) -> str:
    """Mirrors Block 5's run_eval.py - no real Claude call, no token usage
    to report. This repo's own scoring reads total_patients/drug_a_count/
    drug_b_count, never the free-text answer's content, so the scored
    eval doesn't need a real model call."""
    return "Stub answer for CI - not a real model response."


def _make_clinical_agent_fn():
    """Wraps run_agent's search_fn/answer_fn overrides into a single
    clinical_agent_fn closure matching run_multi_agent_async's
    (question) -> (ClinicalAnswer, bool, dict) contract."""
    run_agent_kwargs = {}
    if os.environ.get("USE_RAG_FIXTURES"):
        run_agent_kwargs["search_fn"] = _make_fixture_search_fn()
    if os.environ.get("USE_STUB_ANSWER_FN"):
        run_agent_kwargs["answer_fn"] = _stub_answer_fn

    def _clinical_agent_fn(question: QuestionInput):
        return run_agent(question, **run_agent_kwargs)

    return _clinical_agent_fn


def _question_input(entry: dict) -> QuestionInput:
    return QuestionInput(
        condition=entry["condition"],
        lab=entry["lab"],
        comparison=entry["comparison"],
        value=entry["value"],
        drug_a=entry["drug_a"],
        drug_b=entry["drug_b"],
    )


# --- running all the questions ---------------------------------------------


async def _run_all_questions(questions: list[dict], clinical_agent_fn) -> dict:
    """Runs every question in data/eval/questions.json concurrently via
    asyncio.gather.

    Safe specifically under this CI configuration: with USE_RAG_FIXTURES
    and USE_STUB_ANSWER_FN both set, Role 1 makes zero real external
    calls (search_fn replays a local fixture dict, answer_fn is a stub -
    count_fn is real but hits Neo4j, not a rate-limited API) and Role 2
    only ever hits the local disposable Neo4j container this same CI job
    started - so docs/plan.md §13's real-external-API rate-limit concern
    (up to 2 calls per question if every question ran concurrently
    against real services - 24 simultaneous Pinecone/Claude calls for
    today's 12-question set) does not apply here. A future run against
    the real search service/LLM would need to revisit this decision, not
    assume it still holds.
    """
    coroutines = [
        run_multi_agent_async(_question_input(q), clinical_agent_fn=clinical_agent_fn)
        for q in questions
    ]
    answers = await asyncio.gather(*coroutines)
    return dict(zip((q["id"] for q in questions), answers))


# --- scoring ----------------------------------------------------------------


def _score_answerable_questions(questions: list[dict], answer_key: dict, results: dict) -> dict:
    """Recall over every answerable question (currently 9: q1-q8, q12) -
    exact match on total_patients/drug_a_count/drug_b_count against the
    ground truth.
    """
    scored = []
    discrepancy_ids = []
    for q in questions:
        if not q.get("answerable", True):
            continue
        golden = answer_key[q["id"]]
        answer = results[q["id"]]
        correct = (
            answer.total_patients == golden["total_patients_matched"]
            and answer.drug_a_count == golden["drug_a_count"]
            and answer.drug_b_count == golden["drug_b_count"]
        )
        scored.append({"id": q["id"], "correct": correct})
        if answer.discrepancy_flag:
            discrepancy_ids.append(q["id"])

    correct_count = sum(1 for s in scored if s["correct"])
    recall = correct_count / len(scored) if scored else 0.0
    return {
        "scored": scored,
        "recall": recall,
        "correct_count": correct_count,
        "total": len(scored),
        "discrepancy_ids": discrepancy_ids,
    }


def _check_unanswerable_questions(questions: list[dict], results: dict) -> dict:
    """Pass/fail for q9-q11 - never recall-scored. Pass iff the system
    reports zero matching patients for a condition Block 3's graph has
    never heard of."""
    checks = []
    for q in questions:
        if q.get("answerable", True):
            continue
        answer = results[q["id"]]
        checks.append({"id": q["id"], "passed": answer.total_patients == 0})
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


# --- degradation-matrix eval dimension (fakes, no live calls) --------------


def _fake_clinical_agent_fn(rag_patient_ids, drug_counts, outcome="answered", caveat=None):
    """A minimal clinical_agent_fn fake - same shape as
    tests/test_orchestrator.py's fakes, duplicated here (not imported from
    tests/) so this eval dimension has no dependency on the test suite's
    layout."""
    answer = ClinicalAnswer(
        question="degradation-matrix check question",
        answer="fake answer",
        rag_patient_ids=rag_patient_ids,
        rag_citations=[],
        graph_result=drug_counts,
        confidence="high",
        caveat=caveat,
        outcome=outcome,
    )
    zero_cost = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    count_step_ran = outcome != "nothing_found"
    return lambda question: (answer, count_step_ran, zero_cost)


def _fake_cohort_agent_fn(total, drug_a_count, drug_b_count, outcome="answered"):
    def _fn(question: QuestionInput) -> CohortResult:
        return CohortResult(
            question=question,
            total_patients_matched=total,
            drug_a_count=drug_a_count,
            drug_b_count=drug_b_count,
            patient_ids=list(range(1, total + 1)),
            outcome=outcome,
        )

    return _fn


async def _check_degradation_matrix() -> dict:
    """Injects synthetic tool failures via fakes (spec.md §6's new eval
    dimension) - asserts every row of spec.md §4's matrix still produces
    the correct mode. No live Neo4j/RAG/LLM call in this check."""
    question = QuestionInput(
        condition="hypertension",
        lab="SBP",
        comparison="above",
        value=140,
        drug_a="Lisinopril",
        drug_b="Amlodipine",
    )
    rows = [
        (
            "both_succeed",
            _fake_clinical_agent_fn([1, 2], {"Lisinopril": 1, "Amlodipine": 1}),
            _fake_cohort_agent_fn(2, 1, 1),
            "reconciled",
        ),
        (
            "clinical_tool_error_cohort_succeeds",
            _fake_clinical_agent_fn([], {}, outcome="tool_error"),
            _fake_cohort_agent_fn(5, 2, 1),
            "cohort_only_degraded",
        ),
        (
            "clinical_succeeds_cohort_tool_error",
            _fake_clinical_agent_fn([1, 2], {"Lisinopril": 1, "Amlodipine": 1}),
            _fake_cohort_agent_fn(0, 0, 0, outcome="tool_error"),
            "clinical_only_degraded",
        ),
        (
            "both_tool_error",
            _fake_clinical_agent_fn([], {}, outcome="tool_error"),
            _fake_cohort_agent_fn(0, 0, 0, outcome="tool_error"),
            "both_failed",
        ),
    ]

    row_results = []
    for name, clinical_fn, cohort_fn, expected_mode in rows:
        answer = await run_multi_agent_async(
            question, clinical_agent_fn=clinical_fn, cohort_agent_fn=cohort_fn
        )
        row_results.append(
            {
                "row": name,
                "expected_mode": expected_mode,
                "actual_mode": answer.mode,
                "passed": answer.mode == expected_mode,
            }
        )

    return {"rows": row_results, "passed": all(r["passed"] for r in row_results)}


# --- latency/cost stats from data/eval/run_log.jsonl -----------------------


def _count_existing_log_lines() -> int:
    if not LOG_PATH.exists():
        return 0
    return sum(1 for _ in LOG_PATH.open(encoding="utf-8"))


def _read_log_entries_between(start_line_count: int, end_line_count: int) -> list[dict]:
    """Only the real question runs' own log lines - excludes both any
    entries pytest (or a previous eval run) already wrote to the same
    append-only file earlier in this CI job (before start_line_count),
    and the degradation-matrix eval dimension's later synthetic
    fake-driven entries (after end_line_count), which are near-instant
    and would otherwise dilute the latency/cost baseline."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[start_line_count:end_line_count]]


def _compute_latency_cost_stats(entries: list[dict]) -> dict:
    if not entries:
        return {
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "avg_cost_usd": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
        }
    latencies = sorted(e["latency_ms"] for e in entries)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    total_cost = sum(e["cost_usd"] for e in entries)
    total_tokens = sum(e["tokens"] for e in entries)
    return {
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(latencies[p95_index], 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / len(entries), 6),
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / len(entries), 2),
    }


# --- regression gate ---------------------------------------------------


def _check_regression_gate(recall: float, p95_latency_ms: float) -> dict:
    """Writes data/eval/latency_baseline.json on its first-ever real run;
    every run after compares this run's recall/p95 latency against it.
    Only a recall regression fails the build (matching this project's
    existing regression-gate convention) - the p95 latency comparison is
    reported for visibility, not a hard gate on its own, since no latency
    threshold has been decided anywhere in spec.md/plan.md.
    """
    if not BASELINE_PATH.exists():
        baseline = {"recall": recall, "p95_latency_ms": p95_latency_ms}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        return {"first_run": True, "recall_regressed": False, "baseline": baseline}

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        "first_run": False,
        "recall_regressed": recall < baseline["recall"],
        "baseline": baseline,
        "p95_latency_delta_ms": round(p95_latency_ms - baseline["p95_latency_ms"], 2),
    }


# --- report + CI gate --------------------------------------------------


def _print_report(
    recall_result: dict,
    unanswerable_result: dict,
    degradation_result: dict,
    stats: dict,
    regression: dict,
) -> None:
    print(
        f"Recall ({recall_result['total']} scored questions): {recall_result['recall']:.3f} "
        f"({recall_result['correct_count']}/{recall_result['total']})"
    )
    for s in recall_result["scored"]:
        print(f"  - {s['id']}: {'PASS' if s['correct'] else 'FAIL'}")

    print(f"\nUnanswerable control questions (q9-q11): {'PASS' if unanswerable_result['passed'] else 'FAIL'}")
    for c in unanswerable_result["checks"]:
        print(f"  - {c['id']}: {'PASS' if c['passed'] else 'FAIL'}")

    print(
        f"\nDiscrepancy check ({recall_result['total']} scored questions): "
        f"{'FAIL' if recall_result['discrepancy_ids'] else 'PASS'}"
    )
    if recall_result["discrepancy_ids"]:
        print(f"  discrepancy_flag=True on: {', '.join(recall_result['discrepancy_ids'])}")

    print(f"\nDegradation-matrix check: {'PASS' if degradation_result['passed'] else 'FAIL'}")
    for r in degradation_result["rows"]:
        print(f"  - {r['row']}: expected {r['expected_mode']!r}, got {r['actual_mode']!r} - {'PASS' if r['passed'] else 'FAIL'}")

    print(
        f"\nLatency/cost (this run, from {LOG_PATH}): "
        f"median={stats['median_latency_ms']}ms, p95={stats['p95_latency_ms']}ms, "
        f"total_cost=${stats['total_cost_usd']}, avg_cost=${stats['avg_cost_usd']}, "
        f"total_tokens={stats['total_tokens']}, avg_tokens={stats['avg_tokens']}"
    )

    if regression["first_run"]:
        print(f"\nRegression gate: no baseline yet - wrote {BASELINE_PATH} from this run.")
    else:
        print(
            f"\nRegression gate: baseline recall={regression['baseline']['recall']:.3f}, "
            f"this run={recall_result['recall']:.3f} - "
            f"{'REGRESSED' if regression['recall_regressed'] else 'OK'}. "
            f"p95 latency delta vs baseline: {regression['p95_latency_delta_ms']}ms (reported, not a gate)."
        )


async def run_evaluation() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answer_key = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    clinical_agent_fn = _make_clinical_agent_fn()

    log_start = _count_existing_log_lines()
    results = await _run_all_questions(questions, clinical_agent_fn)
    # Captured here, before the degradation-matrix dimension below runs
    # its own fake-driven invocations and appends its own log entries -
    # only the real question runs' lines fall within [log_start, log_end).
    log_end_of_real_runs = _count_existing_log_lines()
    degradation_result = await _check_degradation_matrix()
    new_entries = _read_log_entries_between(log_start, log_end_of_real_runs)
    stats = _compute_latency_cost_stats(new_entries)

    recall_result = _score_answerable_questions(questions, answer_key, results)
    unanswerable_result = _check_unanswerable_questions(questions, results)
    regression = _check_regression_gate(recall_result["recall"], stats["p95_latency_ms"])

    _print_report(recall_result, unanswerable_result, degradation_result, stats, regression)

    failures = []
    if recall_result["recall"] < RECALL_THRESHOLD:
        failures.append(f"recall {recall_result['recall']:.3f} below threshold {RECALL_THRESHOLD:.2f}")
    if regression["recall_regressed"]:
        failures.append(
            f"recall regressed vs baseline ({regression['baseline']['recall']:.3f} -> {recall_result['recall']:.3f})"
        )
    if recall_result["discrepancy_ids"]:
        failures.append(f"discrepancy_flag=True on: {', '.join(recall_result['discrepancy_ids'])}")
    if not unanswerable_result["passed"]:
        failed_ids = [c["id"] for c in unanswerable_result["checks"] if not c["passed"]]
        failures.append(f"unanswerable control question(s) reported patients: {', '.join(failed_ids)}")
    if not degradation_result["passed"]:
        failed_rows = [r["row"] for r in degradation_result["rows"] if not r["passed"]]
        failures.append(f"degradation-matrix row(s) produced the wrong mode: {', '.join(failed_rows)}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll eval checks passed.")
    return 0


def main() -> None:
    exit_code = asyncio.run(run_evaluation())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
