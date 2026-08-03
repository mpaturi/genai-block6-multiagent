"""Tests for scripts/run_eval.py's log-offset-based latency/cost stats
(see docs/plan.md §9, spec.md §6).

No live Neo4j/RAG/LLM calls - these exercise the pure log-reading/offset
logic against a real temp file standing in for data/eval/run_log.jsonl
(scripts.run_eval.LOG_PATH monkeypatched to point at it).
"""
import json

from scripts import run_eval


def _append_fake_entries(path, count, latency_start=100.0):
    with path.open("a", encoding="utf-8") as f:
        for i in range(count):
            entry = {"latency_ms": latency_start + i, "cost_usd": 0.001, "tokens": 10}
            f.write(json.dumps(entry) + "\n")


def test_degradation_matrix_entries_appended_after_the_real_runs_are_excluded(tmp_path, monkeypatch):
    # Reproduces the real bug this fix targets: run_evaluation() runs the
    # 11 real questions (appending 11 log entries), then the
    # degradation-matrix eval dimension appends 4 more fake-driven entries
    # to the same append-only log afterward. The latency/cost baseline
    # must only ever see the 11 real entries, not all 15 - the fake rows
    # are near-instant and would otherwise dilute the median/p95.
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(run_eval, "LOG_PATH", log_path)

    log_start = run_eval._count_existing_log_lines()
    _append_fake_entries(log_path, 11)
    log_end_of_real_runs = run_eval._count_existing_log_lines()
    _append_fake_entries(log_path, 4)  # simulates the degradation matrix's later writes

    entries = run_eval._read_log_entries_between(log_start, log_end_of_real_runs)

    assert len(entries) == 11


def test_entries_written_before_this_run_started_are_also_excluded(tmp_path, monkeypatch):
    # A previous eval run or pytest already wrote entries to the same
    # append-only file earlier in the same CI job - those must be
    # excluded too, same as before this fix.
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(run_eval, "LOG_PATH", log_path)
    _append_fake_entries(log_path, 3)

    log_start = run_eval._count_existing_log_lines()
    _append_fake_entries(log_path, 11)
    log_end_of_real_runs = run_eval._count_existing_log_lines()
    _append_fake_entries(log_path, 4)

    entries = run_eval._read_log_entries_between(log_start, log_end_of_real_runs)

    assert len(entries) == 11


def test_missing_log_file_returns_no_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "LOG_PATH", tmp_path / "does_not_exist.jsonl")

    assert run_eval._read_log_entries_between(0, 0) == []
