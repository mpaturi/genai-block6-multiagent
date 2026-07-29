"""Writes one log entry per run_multi_agent invocation to
data/eval/run_log.jsonl (see docs/plan.md §9).

Same append-only JSONL pattern as Block 5's block5_agent/logging_utils.py.
This log file is generated output, not source code - it isn't committed
(see .gitignore). Written unconditionally (production and eval runs both
log here), so the eval harness can read real cost/latency back without
re-deriving it from trace data.
"""
import json
import time
import uuid
from pathlib import Path

LOG_PATH = Path("data/eval/run_log.jsonl")


def log_multiagent_run(
    *,
    question: str,
    mode: str,
    confidence: str,
    discrepancy_flag: bool,
    total_patients: int,
    latency_ms: float,
    cost_usd: float,
    tokens: int,
) -> dict:
    """Append one JSON line for this run and return the entry that was written."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": str(uuid.uuid4()),
        "question": question,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "confidence": confidence,
        "discrepancy_flag": discrepancy_flag,
        "total_patients": total_patients,
        "latency_ms": round(latency_ms, 2),
        "cost_usd": round(cost_usd, 6),
        "tokens": tokens,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
