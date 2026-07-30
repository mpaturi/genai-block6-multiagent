# Block 6 — Multi-Agent Clinical Cohort System — Plan

Reads against `docs/spec.md`. Every "decide in plan.md" item spec.md left open gets a concrete decision below, with reasoning — not just a restatement of the question. Where a decision requires a number that can only come from running something against real data (not derivable from reasoning alone), that's flagged as a task for `tasks.md`, not guessed here.

## 1. Cohort Agent entry point

Role 2's entry point returns `CohortResult` alone, not a `tuple[CohortResult, bool]`. Block 5's `run_agent(...) -> tuple[ClinicalAnswer, bool]` shape doesn't carry over: its second value (`count_step_ran`) reports whether Role 1's multi-step pipeline (search → count → synthesize) actually reached its count step — a meaningful signal for a multi-step pipeline. Role 2 has no equivalent structure; it's one combined enumerate-and-count Cypher call, not a pipeline with an optional sub-step to report on. A second boolean here would have no defined meaning.

```python
run_cohort_agent(question: QuestionInput, *, graph_query_fn=query_full_cohort, count_fn=count_drugs_exhaustive) -> CohortResult
```
`docs/spec.md` §2 states this signature directly.

## 2. File layout

**Cross-repo import mechanism:** Block 5 wasn't pip-installable (no `setup.py`/`pyproject.toml`) when this was first written. Resolved by adding minimal packaging to Block 5's own repo (renamed `scripts/` → `block5_agent/`, added `pyproject.toml`, merged to Block 5's `main`) rather than vendoring a copy of its types here or working around it with `sys.path`/`PYTHONPATH` tricks — packaging metadata isn't an interface change, so this doesn't conflict with §8's "modifying Block 5's interfaces happens in its own repo" scoping. `requirements.txt` pins `block5_agent @ git+https://github.com/mpaturi/genai-block5-agent.git@main`; `scripts/schemas.py` imports `QuestionInput`/`ClinicalAnswer` from `block5_agent.schemas` directly.

New repo `genai-block6-multiagent`:
```
scripts/
  schemas.py          # CohortResult, ReconciliationResult, Citation, MultiAgentAnswer, MultiAgentState
  cohort_tool.py       # Cypher query text, exhaustive count query, Neo4j driver factory
  cohort_agent.py      # run_cohort_agent — retry loop around cohort_tool, no internal LangGraph (single step, doesn't need one)
  orchestrator.py       # MultiAgentState graph: dispatch → (clinical_node, cohort_node) → reconcile_node → run_multi_agent entry point
  error_classification.py  # shared helper mapping caught exceptions to Literal["timeout","connection_error","validation_error","unknown"]
  vocabulary_check.py   # startup/CI validation script — see §12
  run_log.py            # append-only JSONL logger — see §9
data/
  eval/
    questions.json      # Block 5's fixed question set plus q12 (§10's seed-extension followup) - 12 total: 9 scored for recall, plus 3 deliberately-unanswerable control questions ("answerable": false) checked pass/fail instead
    answer_key.json      # regenerated — see §11 (ground-truth re-verification)
  seed/
    ci_graph_seed.cypher # same frozen snapshot Block 5's CI uses — see §10
tests/
  test_schemas.py
  test_cohort_tool.py        # Cypher string assertions (parameterized, read-only) + fakes for query/count results
  test_cohort_agent.py       # retry/degrade behavior via fakes
  test_orchestrator.py        # reconciliation logic, all branches of §4's matrix + §2's reconciliation rules, with fakes for both agents
  test_eval_harness.py
docs/
  spec.md
  plan.md
  tasks.md
```

## 3. Graph structure

```
dispatch ──► clinical_node ──┐
         └─► cohort_node   ──┴─► reconcile_node ──► END
```
One dispatch node fans out to two branches; both write to disjoint keys in `MultiAgentState` (`clinical_*` vs `cohort_*`), so no reducer is needed — LangGraph's default merge (union of returned keys) is sufficient since neither branch ever writes a key the other one also writes. `reconcile_node` reads both branches' results/errors and produces `final_answer`.

## 4. Execution model — how "parallel, bounded by the slower branch" actually happens

Block 5's `run_agent` and the new `run_cohort_agent` are both synchronous, blocking functions — that's not changing (Role 1 is reused unmodified, and Role 2's Cypher call is naturally synchronous too). To get real concurrency between the two branches without rewriting either:

- Define `clinical_node` and `cohort_node` as `async def` node functions.
- Inside each, call the underlying synchronous function via a **dedicated** executor — `loop.run_in_executor(block6_executor, run_agent, question, ...)` (respectively `run_cohort_agent`), not bare `asyncio.to_thread`, which always draws from Python's shared default thread pool. `block6_executor` is a `concurrent.futures.ThreadPoolExecutor` this repo owns, sized deliberately (e.g. 4–8 workers is plenty for two branches per question) — see the thread-leak note below for why sharing the default pool is a real risk here.
- Compile the graph normally; invoke it with `await graph.ainvoke(initial_state)`.
- Expose **both** a sync and an async public entry point, not just one: `run_multi_agent(question) -> MultiAgentAnswer` (a thin wrapper around `asyncio.run(graph.ainvoke(...))`) for this repo's own CLI/eval-harness usage, **and** `async def run_multi_agent_async(question) -> MultiAgentAnswer` (directly `await`s `graph.ainvoke(...)`) for callers that are already inside a running event loop. This matters because `asyncio.run()` raises `RuntimeError: asyncio.run() cannot be called from a running event loop` if invoked from inside one — and spec.md §8 already anticipates Block 8 wrapping this in a FastAPI service, whose route handlers calling external APIs are almost always `async def`. Building only the sync wrapper today would work fine for this block's own testing and CI, then break on first contact with the exact integration spec.md flags as the eventual consumer. `run_multi_agent` can simply delegate to `run_multi_agent_async` under the hood (`asyncio.run(run_multi_agent_async(question))`) so there's one real implementation, not two.

This is the concrete mechanism behind "let LangGraph do it natively" — the graph's shape doesn't create concurrency by itself, the executor + `.ainvoke()` does.

**Thread-leak risk, and why it matters here specifically:** `asyncio.wait_for(..., timeout=150)` (§7) stops the *awaiting* coroutine when it times out, but it cannot forcibly kill the underlying OS thread `run_in_executor` started — Python threads aren't cancellable. If `run_agent` or `run_cohort_agent` is genuinely hung (not just slow), that thread keeps running indefinitely in the background even after the branch has been reported as timed out. Two consequences to design around, not just note: (1) if that orphaned call eventually does return, its result is silently discarded — log a warning when this happens (comparing the thread's actual completion time against when the timeout fired) so a CI run doesn't quietly hide "it actually would have succeeded 20s later" information; (2) repeated timeouts (e.g. during a sustained Neo4j outage across many eval questions) leave repeated orphaned threads occupying executor slots — this is exactly why `block6_executor` above is a dedicated pool sized for this repo's own load, not the shared default `asyncio.to_thread` pool every other concurrent asyncio operation in the process also draws from. The 150s ceiling should be understood as a best-effort supervisory backstop for hangs that occur *before* the tool's own internal timeouts even start their clock (e.g. DNS resolution, initial TCP handshake) — the real defense against a hung call is still each tool's own internal timeout (10s Neo4j, 10s Pinecone/Claude), which actually aborts the in-flight request at the client-library level, not just stops waiting for it.

## 5. Node-level defensive error handling

Every node wrapper follows the same shape:
```python
async def clinical_node(state: MultiAgentState) -> dict:
    loop = asyncio.get_running_loop()
    try:
        answer, count_step_ran = await asyncio.wait_for(
            loop.run_in_executor(block6_executor, run_agent, state["question"]),
            timeout=150,
        )
        return {"clinical_result": answer, "clinical_count_step_ran": count_step_ran}
    except Exception as e:
        return {
            "clinical_error": str(e),
            "clinical_error_kind": classify_exception(e),
        }
```
(`block6_executor` is the dedicated `ThreadPoolExecutor` from §4, not the shared default pool — see §4's thread-leak note for why that distinction matters.)
`classify_exception` (in `error_classification.py`) maps caught exception types to the four `Literal` kinds:
- `asyncio.TimeoutError`, `neo4j.exceptions.ServiceUnavailable` (on connection-level failures with a timeout component) → `"timeout"`
- `neo4j.exceptions.ServiceUnavailable` (connection refused, not timed out), `ConnectionError`, `httpx.ConnectError` → `"connection_error"`
- `pydantic.ValidationError` → `"validation_error"`
- anything else → `"unknown"`

This is a genuine second line of defense: `run_agent`/`run_cohort_agent` are trusted to never raise on their own documented failure modes, but the node wrapper's `try/except` catches anything outside that contract (e.g. a bug in a test fake, a serialization failure) so it degrades into `mode="both_failed"` territory instead of crashing the whole graph invocation.

## 6. Neo4j driver lifecycle and permissions

- **Lifecycle:** one `neo4j.Driver` instance constructed at process/test-session start (a small `get_driver()` factory, cached via `functools.lru_cache(maxsize=1)`), injected into `run_cohort_agent` as a default keyword argument (same DI pattern as `graph_query_fn`/`count_fn`) so tests can swap in a fake driver or a test-container instance without touching production code. Never construct a fresh driver per call.
- **Teardown:** nothing currently closes this driver — needs an explicit path, not an assumption it'll be fine. In tests, a session-scoped pytest fixture calls `driver.close()` after the full test session finishes, so a disposable Neo4j test container gets a clean disconnect before teardown rather than risking flaky CI from lingering connections. For any future long-running deployment (Block 8), a real shutdown hook (e.g. on process signal) needs to call `.close()` too — noted here so it isn't silently missing once this stops being a short-lived CI process.
- **DB-level permissions:** create a dedicated Neo4j role/user for this repo (e.g. `block6_readonly`) scoped to read-only privileges (`MATCH`/traversal, no write privileges) on the patient/condition/drug labels this system touches — distinct from whatever credentials Block 3 uses to load data. This is defense-in-depth alongside the application-level read-only Cypher constraint from spec.md §2; a task in `tasks.md` covers actually provisioning this role, since it depends on the real Neo4j instance's admin access, not something plan.md can do abstractly.
  **Phase 4 finding: not buildable on this stack.** Every block in this project runs Neo4j Community Edition (`neo4j:5.18-community`, per Block 3's `docker-compose.yml`), and Community Edition has no role-based access control at all — `CREATE ROLE`/`GRANT`/`SHOW ROLES` are Enterprise-only administration commands. Confirmed directly against a disposable instance of this exact image: `SHOW ROLES` fails with `Unsupported administration command`, and `CALL dbms.components()` reports `edition: "community"`. There is no `block6_readonly` role to provision, and no lesser substitute either — Community Edition's only other users get full, unrestricted access, so creating one wouldn't be a read-only boundary, just a second name/password pair with the same privileges as the driver's own credentials. **The application-level read-only Cypher constraint (parameterized, `MATCH`/`RETURN`-only, built and tested in Phase 3) is therefore the sole enforced defense for now** — not one layer of two, until this stack is ever run on Neo4j Enterprise Edition or an equivalent managed offering with RBAC support. Tracked as a "what I'd do next" item in `tasks.md` Phase 6, not a Phase 4 oversight.

## 7. Timeouts

Derived from confirmed constants, not guessed round numbers:

- **Cohort Agent's Cypher call:** starts at **10s**, matching Block 5's existing `count_drugs` Neo4j timeout precedent (the only comparable Neo4j-timeout data point this project has). This is a starting value, not a permanently fixed one — the unbounded query (no `person_id IN [...]` pre-filter) may scan more of the graph than the query it's derived from. If eval runs show it's consistently too tight, raise it and record the real number that worked, not silently bump it.
- **Branch-level supervisory timeout:** derived from Role 1's own worst case, since it's the slower agent. `_MAX_TOOL_RETRIES = 2` (3 attempts) on search, 3 attempts on count (10s Neo4j timeout each ⇒ ~30s worst case), and `_MAX_ANSWER_RETRIES = 1` (2 attempts) on answer synthesis (~10s/attempt ⇒ ~20s worst case).

  **Search step:** Block 5's own outbound HTTP client to Block 4's `/query` (`rag_tool.py:60`) sets `requests.post(..., timeout=10)` — tighter than Block 4's internal 10s+10s ceiling, not looser or equal. The client-side call aborts at 10s regardless of whether Block 4 is still working server-side, so the real worst case per search attempt is **10s** ⇒ **~30s worst case for 3 attempts**.

  Sum: 30s (search) + 30s (count) + 20s (synthesis) = **~80s theoretical worst case for Role 1**.

  Setting the branch-level ceiling below the worst case would cut off legitimate retries the tool's own logic is still working through — so the ceiling exists only to catch hangs *outside* what these timeouts already cover (e.g. a hung DNS resolution or TCP handshake before a client library's own timeout clock even starts). **Decision: 150s per branch** — nearly 2x the ~80s worst case, generous headroom without needing to re-tune it to shave off margin that costs nothing to keep. Applied uniformly to both branches (Role 2's own worst case is much shorter — 3 attempts × 10s ⇒ ~30s). Enforced via `asyncio.wait_for(...)` wrapping each `loop.run_in_executor(block6_executor, ...)` call in §4/§5's node wrappers; a `TimeoutError` here is caught by §5's node-level try/except and classified as `"timeout"` — with the caveat from §4's thread-leak note that this stops the wait, not the underlying thread.

## 8. Confidence tier redesign

Spec §2 flagged that reusing Block 5's raw thresholds (`high` ≥25) makes `high` trivially achievable once Role 2 removes the ceiling. Rather than picking new arbitrary numbers (which would just be a different arbitrary ceiling), redefine what "high confidence" means for this system — tied to *how the number was obtained*, not *how big it is*. The first draft of this redesign only explicitly covered `mode="reconciled"` and `clinical_only_degraded`, leaving `cohort_only_degraded` uncovered and a gap inside `clinical_only_degraded` itself (a Role 1 count that happened to sit exactly at its old `top_k=25` ceiling didn't fit `medium`'s "15–24" or `low`'s "fewer than 15"). Corrected, exhaustive partition across all four `mode` values:

- **`high`**: Role 2's exhaustive enumeration is the source of the reported count — this covers `mode="reconciled"` (whether or not `total_patients_matched > 25`) *and* `mode="cohort_only_degraded"` (Role 2 is the *only* source there, and it's exhaustive by construction regardless of Role 1's availability). Exhaustive enumeration is the strongest evidence this system can produce either way, so both modes earn it.
- **`medium`**: `mode="clinical_only_degraded"` and Role 1's own patient count (`len(clinical_result.rag_patient_ids)`) is **≥ 15** — broadened from Block 5's original "15–24" band on purpose: without Role 2 confirming it, even a Role-1 count that saturated its old `top_k=25` ceiling is still just Role 1's own best effort, not a confirmed exhaustive count, so it belongs in the same degraded-but-substantial bucket as 15–24, not in `high`.
- **`low`**: `mode="clinical_only_degraded"` with `len(clinical_result.rag_patient_ids)` **< 15**, any `discrepancy_flag=True` case, or `mode="both_failed"`.

Every `MultiAgentAnswer.mode` value now maps to exactly one rule above — worth a unit test per mode in Phase 2 asserting this explicitly (see `tasks.md`), not just inferring it from the mode assertion. This is implemented in the reconcile node, not as a standalone recalibration of Block 5's existing tiers (which stay untouched inside `ClinicalAnswer` — this repo only defines how `MultiAgentAnswer.confidence` gets set, composing but not modifying Block 5's schema per spec §3).

## 9. Observability export

In addition to LangSmith tracing (unchanged from spec §6), `run_log.py` appends one JSON line per `run_multi_agent` invocation to `data/eval/run_log.jsonl`: `{question, mode, confidence, discrepancy_flag, total_patients, latency_ms, cost_usd, tokens}`. This gives Block 8's future cross-system dashboard a queryable source that doesn't require hitting LangSmith's API directly — a plain file this repo already controls. Written unconditionally (production and eval runs both log here); the eval harness reads it back to report per-run cost/latency without re-deriving it from trace data.

## 10. Eval reproducibility

The eval harness runs against the same frozen `ci_graph_seed.cypher` snapshot Block 5's CI already loads (`data/seed/ci_graph_seed.cypher`, originally copied verbatim from Block 5's current file) — loaded fresh into a disposable Neo4j instance per CI run (same seed-and-reset pattern Block 5 established), not a live or shared graph. This directly addresses the "moving graph" risk spec §6 flagged: the enumeration query's result is only meaningful if the population it's enumerating is fixed and known.

**Deliberately extended, no longer identical to Block 5's file:** a PR #5 review found that no combination of condition/lab/comparison/value in Block 5's original seed produces a cohort exceeding the old `top_k=25` ceiling with a non-zero real drug count — every question that could exceed 25 (`Streptococcal pharyngitis`, 26 patients total) has essentially no real prescriptions in that cohort, and every cohort with real prescription data tops out at or under 25. That left `mode="reconciled"`'s `total_patients_matched > 25` reconciliation path (spec §2, plan §8's `high`-confidence rule) recall-scored only by q1/q7, both of which sit exactly at the 25-patient boundary rather than exceeding it — never exercised against a real, non-vacuous drug count. Rather than accept that gap or fabricate ground truth, the seed was extended with a new, fully isolated population for q12 (docs/tasks.md Phase 5 followup): 30 new patients (person IDs 990001–990030, disjoint from every existing ID including patient 900001's untouched hand-written regression case), one new condition (`Chronic kidney disease`, not used by any other question) so no existing question's cohort membership changes, and prescriptions built from Drug nodes already in the seed (`Amlodipine`, `Warfarin` — no new Drug node introduced solely for this case). Re-verified by hand via the real enumeration/drug-count queries against the extended seed, same discipline as Phase 4's q1/q7 re-verification, not assumed from how the patients were constructed: `total_patients_matched=30`, `drug_a_count=15`, `drug_b_count=5`. All 11 original questions' ground truth re-confirmed unchanged against the extended seed before this landed.

## 11. Ground-truth re-verification for the two previously-capped questions

This can't be resolved by reasoning — it requires actually running the unbounded enumeration query against the seed data once, by hand, and recording the real total. **Task, not a plan-level decision** (see `tasks.md` Phase 4). This plan and `docs/tasks.md` do not claim a specific target recall number for those two questions beyond "the exhaustive count, whatever it turns out to be" — recall improving is only a real, checkable claim once that ground truth exists independently of the process being fixed.

## 12. Vocabulary consistency check

Two layers, not one — a CI-time check alone only covers the scored fixed eval questions, but Role 2 runs unconditionally on *every* question this system ever receives (§Role 2 applicability, spec §2), including ones outside the eval set once Block 8 wires this to real input. A check that only runs once against a fixed list doesn't protect that general case.

- **CI-time (`vocabulary_check.py`):** run once before the eval suite, queries Block 3's graph for its distinct `condition_name` values and lab property names, cross-references them against the exact strings in `data/eval/questions.json`. Fails loudly (non-zero exit, names the mismatched string) if any answerable question doesn't have an exact match. `data/eval/questions.json`'s 3 deliberately-unanswerable control questions (`"answerable": false`) are skipped by this check on purpose — their whole point is a condition Block 3's graph has never heard of (e.g. "schizophrenia"), so a mismatch there is the correct, expected outcome, not vocabulary drift to flag.
- **Runtime (`get_known_vocabulary()`):** the same lookup logic, refactored into a reusable function called from `reconcile_node` whenever it hits a `nothing_found`/`answered` split. Instead of writing a generic "this usually means a vocabulary mismatch" guess into `ReconciliationResult.notes`, the reconcile node checks whether the question's actual `condition`/`lab` value is in the known-vocabulary set and states the real finding — "confirmed: `condition` value not present in Block 3's graph" vs. "vocabulary looks consistent; disagreement is unexplained by this check" — for any question, not just the ones scored in the eval set.

This closes the gap between what spec §2's reconciliation logic assumes (shared vocabulary) and what's actually verified — a one-time CI check on a fixed list doesn't cover a question nobody's seen yet.

**Cache staleness for long-running processes:** "cache once per process" is fine for this block's own short-lived eval/CLI runs, but it's a real correctness bug waiting to happen once this becomes a long-running deployed service (Block 8) — a condition or lab added to Block 3's graph after process startup would never enter the cache, and a genuinely new, correct value would get misreported as a vocabulary mismatch. Decision: give the cache a TTL (e.g. re-query Block 3 if the cached vocabulary is more than 15 minutes old) rather than caching indefinitely — cheap enough to check on each `nothing_found`/`answered` split (which should be rare), and it removes the silent-staleness failure mode entirely.

## 13. Latency budget — expected case, not just worst case

Spec §6 asked for both an expected and a worst-case number; §7 above only derived the worst case (to size the branch-level timeout). Estimated expected/happy-path latency, reasoned from the same confirmed timeout ceilings rather than measured (flagged accordingly): Role 1's happy path (one search attempt, one count attempt, one answer-synthesis attempt, no retries) is well under each step's individual timeout ceiling in the normal case — realistically single-digit seconds per step, so an estimated **~5–12s total** for Role 1. Role 2's happy path (one Cypher query against an indexed graph) is realistically **under 2s**. Since both run in parallel, expected total latency ≈ Role 1's happy path, **~5–12s per question**.

This is an estimate for planning purposes only, not a measured number — flagged explicitly per this project's "real measured numbers only" convention. Phase 5 (`tasks.md`) must record the actual measured median/p95 latency from real CI eval runs, and that measured number — not this estimate — is what the eval harness's regression check should compare future runs against.

**Flagging, not deciding here — depends on how the eval harness in Phase 5 actually gets written:** if all fixed questions run concurrently (not just each question's own internal two-agent fan-out, but multiple questions in flight at once), that's up to 2 real external API calls per question — 24 simultaneous calls to real external APIs (Pinecone, Claude) with real rate limits for today's 12-question set — Role 2 runs unconditionally on every question, including the 3 deliberately-unanswerable ones, so they count toward this too — on top of the thread-pool sizing concern above. Whoever writes Phase 5's eval harness should decide explicitly whether questions run sequentially or concurrently — not default into concurrency by accident just because `asyncio.gather` is easy to reach for once the per-question fan-out is already async.

## 14. Discrepancy handling in the automated eval suite

The scored fixed questions (Block 5's original 8, plus q12 added in Phase 5 — see §10) have no eval question that's expected to produce a `discrepancy_flag=True` result — all are real clinical questions with a single correct answer, not deliberately seeded vocabulary-mismatch tests. So: **any `discrepancy_flag=True` result on any scored question is treated as an eval failure**, reported separately from the recall metric (so it's clear *why* it failed — a disagreement, not a wrong number) but still failing the CI gate. If a future eval question is deliberately added to exercise the discrepancy path itself (e.g. a synthetic vocabulary-mismatch case), that specific question's discrepancy would be asserted as the expected outcome — but that's new eval-set content, not something this phase needs to add.

## 15. Answer-text synthesis for `cohort_only_degraded`

Hardcoded template, not a fallback LLM call — decided for two reasons: it keeps this degraded path fully deterministic in CI (an LLM call here would reintroduce the nondeterminism/cost Role 1's own failure was supposed to remove from the equation), and Role 1's LLM is precisely the thing that just failed in this mode, so calling out to another LLM to describe that failure adds a second point of failure to a path whose entire purpose is graceful degradation. Template:
```python
f"Of {r.total_patients_matched} patients with {q.condition} and {q.lab} {q.comparison} {q.value}, "
f"{r.drug_a_count} are on {q.drug_a} and {r.drug_b_count} are on {q.drug_b}. "
f"No supporting evidence citations are available for this run because the clinical evidence agent failed."
```

## 16. PHI in traces

Patient data across Blocks 3–6 is synthetic/de-identified, not real PHI. Full state (including patient IDs and RAG-cited snippets) is accepted as-is in LangSmith traces for this block — no redaction needed.

## 17. Cost of always running both agents

Accepted as the point of the exercise — the assignment is specifically about reconciling two agents, and short-circuiting one based on the other's result would undercut the "coordinate to complete a task neither does alone" requirement itself. Noted as a "what I'd do next" candidate (mirroring Block 5's own pattern) rather than built now: a cheap pre-check (e.g. run Role 2 first, since it's fast, and only invoke Role 1's expensive LLM path when citations are actually needed for the final answer) is a real future optimization, explicitly deferred.

## 18. Testing strategy

TDD: write failing tests first against the schemas and reconciliation logic (using fakes for `run_agent`, `run_cohort_agent`, and the Neo4j driver — never live calls in unit tests), covering every row of spec §4's matrix, every bullet of §2's reconciliation rules (including the `nothing_found`/`answered` split and the `clinical_count_step_ran=False` gating), and the node-wrapper try/except behavior from §5 above. Only after those tests exist and fail for the right reason does implementation begin.
