# genai-block6-multiagent

Two-agent LangGraph system reconciling RAG-grounded and exhaustive graph-enumeration answers to clinical cohort questions.

## The problem

Block 5 shipped a single clinical QA agent that answers "of patients with [condition] and [lab] [comparison] [value], how many are on [drug_a] vs [drug_b]?" It measured mean recall 0.789 across 8 questions, with 2 of 8 structurally capped: its drug-count step only ever sees the patient IDs its RAG search step handed it (`top_k` ≤ 25), so any real cohort larger than 25 is undercounted by construction — a retrieval ceiling, not a retrieval failure. Block 5's own spec flagged the fix as future work: enumerate the cohort directly from the graph, with no ceiling.

Block 6 builds that fix as a second, distinct agent, and makes the *combination* of the two agents the actual deliverable — neither agent alone produces an answer that's both **exhaustive** (correct count, no ceiling) and **evidence-grounded** (cited patient records). Block 5's agent has citations but a capped count; the new agent has an exhaustive count but no citations. An orchestrator reconciles them.

## Architecture

```
dispatch ──► clinical_node ──┐
         └─► cohort_node   ──┴─► reconcile_node ──► END
```

- **`clinical_node`** wraps Block 5's `run_agent` unmodified (RAG search → graph-verified drug count → LLM-written answer), run in a dedicated thread pool so it doesn't block the event loop.
- **`cohort_node`** wraps this repo's own `run_cohort_agent` — one parameterized, read-only Cypher call that enumerates *every* matching patient directly from Neo4j and counts both named drugs over the full cohort. No RAG, no `top_k`, no citations.
- Both branches run concurrently (via `asyncio.wait_for` + a dedicated `ThreadPoolExecutor`, not bare `asyncio.to_thread`), each under a 150s supervisory timeout, and write to disjoint keys in one explicit, inspectable `MultiAgentState` — no hidden state, no reducer needed.
- **`reconcile_node`** compares both results: matching counts under 25 patients are `mode="reconciled"`/`confidence="high"`; a cohort over 25 uses the exhaustive count as authoritative; a real disagreement sets `discrepancy_flag=True` and surfaces both numbers rather than silently picking one; either agent failing degrades gracefully to `cohort_only_degraded`/`clinical_only_degraded`/`both_failed` — the system never raises.

## Tech stack

- **LangGraph** — the orchestrator's `StateGraph`
- **Neo4j** — the Cohort Agent's direct graph queries (reusing Block 3's schema/conventions)
- **Anthropic Claude** — via Block 5's `run_agent`, for the RAG-grounded answer's free-text synthesis
- **Pinecone** (via Block 4/Block 5) — the RAG search step's vector index
- **pytest** — TDD throughout; fakes for every external dependency, no live calls in the test suite
- **GitHub Actions** — CI against a disposable Neo4j service container

## Results (real, measured — Phase 7's actual CI-equivalent run, seed corrected)

Run locally against a fresh, cold-start disposable Neo4j container, in the exact sequence CI uses (seed load → vocabulary check → `pytest` → eval harness), with `USE_RAG_FIXTURES=1 USE_STUB_ANSWER_FN=1` (no real Pinecone/Claude calls — see the cost caveat below):

| Metric | Result |
|---|---|
| Recall (8 scored questions) | **1.000 (8/8)** |
| q1 (previously top_k-capped; seed corrected in Phase 7) | **Correct** — `total_patients=99, drug_a_count=49, drug_b_count=28`, matching Phase 7's independently-verified exhaustive ground truth |
| q7 (previously top_k-capped) | **Correct** against Phase 4's independently-verified exhaustive ground truth (25/25 patients) |
| q9–q11 (deliberately-unanswerable controls) | **All pass** — zero patients correctly reported for all three |
| Discrepancy check (8 scored questions) | **Pass** — no `discrepancy_flag=True` |
| Degradation-matrix check | **Pass** — all 4 rows of the failure matrix produce the correct `mode` |
| Latency (cold start; baseline reset in Phase 7) | median **922ms**, p95 **1031ms** |
| Cost / tokens | $0.0 / 0 tokens |

**q1, before vs. after — the real story, not the coincidence:** Phase 4's first pass at q1 found its true population was exactly 25 patients, identical to Block 5's original RAG-capped count — a correct result at the time, but it turned out to be an artifact of the CI seed itself: Block 5 later discovered (own repo, `phase-12-fix-q1-seed`) that this bucket had only ever been seeded with the exact 25 patients its RAG search returns, so the golden answer and the capped output could never have disagreed, no matter how large the true population really was. With the seed corrected to its true, exhaustive 99-patient population, the contrast is now real and demonstrated, not theoretical: **Block 5's own RAG search still returns only 25 of those 99 patients**, so Block 5 now fails this question's accuracy check permanently, by its own design. **Block 6's Cohort Agent enumerates all 99 directly from the graph**, reconciling to `mode="reconciled"`, `confidence="high"`, the correct 99/49/28 — this is the gap Block 6 exists to close, shown against real data rather than asserted from a seed too small to test it.

**q7, before vs. after:** this one's true population genuinely is 25 — RAG's cap and reality coincide here, and that's still independently confirmed, not assumed.

**Cost caveat:** the $0.0/0-token figures above reflect this CI configuration, where the answer-writing step is stubbed and never calls a real LLM — they are not a real production cost estimate. Separately, the underlying $/token rate this repo's logging inherits from Block 5 is not verified against Anthropic's current published pricing (see "What I'd do next" below) — treat any non-zero `cost_usd` this system reports as directional, not a budget number, until that's checked.

**Latency caveat:** the ~5x latency drop between Phase 5's original baseline (p95 5282ms) and this run (p95 1031ms) is *not* explained by q1's population growing from 25 to 99 patients — a controlled A/B check (same fresh-container methodology, old seed vs. new seed) found both land in the same ~1.0-1.4s p95 range, consistent with Neo4j's relationship traversal making 25 vs. 99 matching rows a non-issue either way. Phase 5's 5282ms was most likely a one-off cold-start outlier for that specific run, not a stable number — treat single-sample latency baselines like this one as noisy until several real runs establish a trend, not as a precise figure on their own.

## AI-assisted workflow

Built with [Claude Code](https://claude.com/claude-code), following spec-driven development throughout: `docs/spec.md` (what and why) → `docs/plan.md` (concrete decisions for everything the spec left open) → `docs/tasks.md` (checklist form of the plan) → code — each phase committed and reviewed before the next began. Each phase lived on its own branch (`phase-1-spec` through `phase-6-docs`), with its own PR against the previous phase's branch, so the sequence of decisions stays legible in the git history rather than arriving as one large, undifferentiated diff. TDD throughout Phase 2/3: every test was written and confirmed failing for the right reason before the implementation that makes it pass.

See `docs/spec.md`'s "What I'd do next" section for known gaps and deferred work.
