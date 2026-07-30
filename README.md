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

## Results (real, measured — Phase 5's actual CI-equivalent run, including its post-review followups)

Run locally against a fresh, cold-start disposable Neo4j container, in the exact sequence CI uses (seed load → vocabulary check → `pytest` → eval harness), with `USE_RAG_FIXTURES=1 USE_STUB_ANSWER_FN=1` (no real Pinecone/Claude calls — see the cost caveat below):

| Metric | Result |
|---|---|
| Recall (9 scored questions) | **1.000 (9/9)** |
| q1 / q7 (previously top_k-capped) | **Both correct** against Phase 4's independently-verified exhaustive ground truth (25/25 patients each) |
| q12 (new — see below) | **Correct** — 30/15/5, independently re-verified by hand against the extended seed |
| q9–q11 (deliberately-unanswerable controls) | **All pass** — zero patients correctly reported for all three |
| Discrepancy check (9 scored questions) | **Pass** — no `discrepancy_flag=True` |
| Degradation-matrix check | **Pass** — all 4 rows of the failure matrix produce the correct `mode` |
| Latency (post-fix; `data/eval/latency_baseline.json`) | median **898.5ms**, p95 **953.0ms** |
| Cost / tokens | $0.0 / 0 tokens |

**q1/q7, before vs. after:** under Block 5 alone, these two questions' true patient counts could never be verified beyond whatever RAG's `top_k=25` happened to retrieve — the reported count and the retrieval ceiling were the same number by construction, so "correct" was unfalsifiable. Block 6's Cohort Agent enumerates them exhaustively instead; for this seed data both true counts turn out to be exactly 25 (identical to Block 5's original numbers), but that's now an independently confirmed fact, not an assumption baked into the measurement.

**q12 — closing a real coverage gap a PR review caught:** q1 and q7 both sit *exactly* at the old 25-patient boundary, so neither actually exercises the `total_patients_matched > 25` reconciliation path against a real, non-zero drug count — and no combination in Block 5's original 11-question set does either (the one cohort that can exceed 25, 26 patients, turns out to have essentially no real prescriptions in it). Rather than leave that gap or fabricate ground truth, `data/seed/ci_graph_seed.cypher` was deliberately extended with an isolated 30-patient cohort (new condition, new patient IDs, existing Drug nodes only — no other question's data touched), independently re-verified by hand the same way as q1/q7: `total_patients_matched=30, drug_a_count=15, drug_b_count=5`.

**Latency, corrected:** the originally-committed baseline (median 5047ms, p95 5282ms) included two distortions since fixed — a degradation-matrix log-dilution bug (the eval harness's synthetic fake-driven runs were mixing their near-instant timings into the same stats as the real question runs) and normal cold-start/machine variance. The number above is the current, corrected measurement.

**Dependency note:** `requirements.txt` briefly pinned `block5_agent` to a Block 5 review branch (`phase-11-expose-cost`) while its `cost_info`-exposing change was in review; that PR has since merged to Block 5's `main` and the pin has been flipped back to `@main`, matching this repo's own convention of never routing around another block's interfaces.

**Cost caveat:** the $0.0/0-token figures above reflect this CI configuration, where the answer-writing step is stubbed and never calls a real LLM — they are not a real production cost estimate. Separately, the underlying $/token rate this repo's logging inherits from Block 5 is not verified against Anthropic's current published pricing (see "What I'd do next" below) — treat any non-zero `cost_usd` this system reports as directional, not a budget number, until that's checked.

## AI-assisted workflow

Built with [Claude Code](https://claude.com/claude-code), following spec-driven development throughout: `docs/spec.md` (what and why) → `docs/plan.md` (concrete decisions for everything the spec left open) → `docs/tasks.md` (checklist form of the plan) → code — each phase committed and reviewed before the next began. Each phase lived on its own branch (`phase-1-spec` through `phase-6-docs`), with its own PR against the previous phase's branch, so the sequence of decisions stays legible in the git history rather than arriving as one large, undifferentiated diff. TDD throughout Phase 2/3: every test was written and confirmed failing for the right reason before the implementation that makes it pass.

See `docs/spec.md`'s "What I'd do next" section for known gaps and deferred work.
