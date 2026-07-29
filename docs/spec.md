# Block 6 — Multi-Agent Clinical Cohort System — Spec

Repo: `genai-block6-multiagent` (new, per portfolio strategy: one repo per block).

**A note on scope:** a few paragraphs below (the orchestrator's execution-model discussion, the Neo4j driver-lifecycle note, specific timeout mechanics) are plan-level "how to build it" detail, not spec-level "what and why" — see `docs/plan.md` for the concrete decisions rather than duplicating them here.

## 1. Problem

Block 5 shipped a single clinical QA agent that answers "of patients with [condition] and [lab] [comparison] [value], how many are on [drug_a] vs [drug_b]?" It measured mean recall 0.789 across 8 questions, with 2 of 8 structurally capped: `count_drugs` only ever sees the patient IDs `search_patients` (RAG, `top_k` ≤ 25) handed it, so any cohort larger than 25 is undercounted by construction, not by retrieval failure. Block 5's spec explicitly deferred the fix: "for questions fully answerable via structured filters alone, the graph could enumerate every matching patient directly with no ceiling" — flagged as Block 6 candidate work.

Block 6 builds that fix as a second, distinct agent, and makes the *combination* of the two agents the actual deliverable: neither agent alone produces a cohort answer that is both **exhaustive** (correct count, no ceiling) and **evidence-grounded** (cited patient records). The Block 5 agent has citations but a capped count; the new agent has an exhaustive count but no citations. An orchestrator reconciles them.

## 2. Roles

**Role 1 — Clinical QA Agent (reused, never edited in this repo).**
Block 5's `run_agent(question, *, search_fn, count_fn, answer_fn) -> tuple[ClinicalAnswer, bool, dict]`, imported as-is from the Block 5 package. The third return value, `cost_info` (`{"cost_usd", "input_tokens", "output_tokens"}`), was added in Block 5's own repo (its `phase-11-expose-cost` branch) during Block 6's Phase 5 — a real interface change, made where §8 says interface changes belong, not a modification made here. This repo threads it through as `MultiAgentState.clinical_cost_info` for `scripts/run_log.py`. Its value: RAG-grounded patient identification (`rag_patient_ids`) plus a verified drug-count breakdown (`graph_result`), capped at `top_k=25`.

**`Citation.snippet`'s data source:** `ClinicalAnswer` exposes `rag_citations: list[dict]`, each entry `{"patient_id": int, "chunk_id": ..., "snippet": str}`, built by `scripts/schemas.py::build_rag_citations()` and populated in `rag_tool.py::search_patients()` from Block 4's per-source `chunk_text`. This is Block 6's `Citation.snippet` data source — see §3.

**Role 2 — Cohort Enumeration Agent (new).**
Queries Neo4j directly via parameterized Cypher on `condition`, `lab`, `comparison`, `value` to enumerate *every* matching patient — no `top_k` ceiling — then counts `drug_a` vs `drug_b` over the full cohort. No RAG call, no citations. Its value: a correct total when the cohort exceeds 25.

**Runs unconditionally for every question, not gated by question type.** Block 5's deferred note (§1) scoped this idea to "a specific subset of questions... fully answerable via structured filters alone" — implying not every question would qualify. But `QuestionInput` (`condition`, `lab`, `comparison`, `value`, `drug_a`, `drug_b`) is *always* fully structured in this system; there's no free-text-only question variant either block handles. So for Block 6, that "subset" is the entire set of questions it will ever receive, and Role 2 runs on all of them unconditionally. Stating this explicitly rather than leaving it implicit, because it stops being true the moment a future block (e.g. Block 8's broader capstone) introduces a question type Role 2 can't safely answer this way — that gating decision would need to be added then, not assumed already handled here.

Entry point (mirrors Block 5's DI pattern for testability with fakes). Unlike Block 5's `run_agent`, this returns `CohortResult` alone, not a `tuple[CohortResult, bool]` — Role 2 is a single combined enumerate-and-count Cypher call, not a multi-step pipeline with an optional sub-step to report on, so a second boolean would have no defined meaning here (see `docs/plan.md` §1):
```
run_cohort_agent(question: QuestionInput, *, graph_query_fn=query_full_cohort, count_fn=count_drugs_exhaustive) -> CohortResult
```
Never raises on tool failure — same retry/degrade contract as Block 5 (2 retries on the Cypher call, matching `_MAX_TOOL_RETRIES = 2` at `agent.py:33`).

**Starting point, don't write from scratch:** Block 5's `graph_tool.py:41-46` already has a `VERIFY_PATIENTS_QUERY_TEMPLATE` — `MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition {{condition_name: $condition}}) WHERE p.{lab_property} {op} $value AND p.person_id IN $person_ids RETURN collect(DISTINCT p.person_id)`. Dropping just the trailing `AND p.person_id IN $person_ids` clause and running the rest of that exact query against the full patient population *is* the unbounded enumeration query this role needs — a one-clause edit of an existing, already-parameterized query, not new Cypher. Reuse the same relationship names it depends on (`HAS_CONDITION`, `PRESCRIBED` — also used identically in Block 3's `query_graph.py:21-29`) and the same driver/env-var convention (`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/`NEO4J_DATABASE`, matching `graph_tool.py:24-27` and Block 3's `check_connection.py:1-28`) so the Cohort Agent's Cypher stays consistent with both repos it sits between.

**Driver lifecycle constraint:** the Neo4j driver must be a long-lived, reused connection, not reconstructed per query — see `docs/plan.md` §6 for the concrete decision (a shared, injectable driver instance) and the reasoning behind it.

**Security constraints (flagging now for Block 7's threat model):**
- Cypher must be built with driver-level query parameters, never string interpolation of `condition`/`lab`/`drug_a`/`drug_b` values — these fields trace back to user-facing input once Block 8 integrates this into the capstone's app layer.
- The Cohort Agent's Cypher must be read-only (`MATCH`/`RETURN` only) — no `CREATE`/`MERGE`/`DELETE`/`SET`. This is a new code path with direct, unmediated access to a graph DB other blocks also depend on; it has no business need to write anything, so it shouldn't be able to.
- Application-code read-only (above) was meant to be one layer, not the whole defense, backed by a DB-level read-only `block6_readonly` role/user — see `docs/plan.md` §6. Phase 4 found this second layer isn't buildable on this project's actual stack (Neo4j Community Edition has no RBAC support at all), so application-code read-only is, for now, the sole enforced defense, not one of two.
- Inherited, not introduced here: Role 1's LLM reads raw patient note text via RAG, and that text flows into `MultiAgentAnswer.answer` with no sanitization step. This is an indirect-prompt-injection surface (OWASP LLM Top 10) that predates Block 6, but Block 6 is the first place it feeds into a reconciled, multi-source answer — worth a paper trail here so Block 7 doesn't discover it cold.

**Orchestrator (LangGraph `StateGraph`).**
Public entry point — the actual "task neither agent does alone" deliverable Block 8 will import:
```
run_multi_agent(question: QuestionInput, *, clinical_agent_fn=run_agent, cohort_agent_fn=run_cohort_agent) -> MultiAgentAnswer
```
Runs both agents via LangGraph's native fan-out — two edges out of a single dispatch node, one to each agent, joined at a reconcile node — not hand-managed asyncio. This gives a genuine latency win (bounded by the slower branch, not the sum of both) with the concurrency mechanics handled declaratively by the framework, and is the more instructive orchestration artifact for this block. Sequential execution remains an acceptable, simpler fallback if a first cut needs to ship fast — losing nothing conceptually — but should still route through LangGraph edges, not raw asyncio, either way.

Two timeout gaps to close, not assumed: the Cohort Agent's Cypher call needs its own explicit timeout (not inherited from the bounded query it's derived from — dropping `person_id IN $person_ids` likely changes its performance profile), and each parallel branch needs a hard ceiling at the graph level, not just reliance on each tool's own internal timeout — see `docs/plan.md` §7 for the specific values and the worst-case reasoning behind them.

**Execution-model constraint:** "bounded by the slower branch, not the sum" only holds if the two branches actually run concurrently — drawing two edges in the graph doesn't guarantee that on its own for synchronous, blocking code (which is what both agents are). See `docs/plan.md` §4 for the concrete mechanism chosen.

**Node wrappers need their own defensive try/except, as a second line of defense**, beyond `run_agent`/`run_cohort_agent`'s own internal "never raise" contracts (§7) — see `docs/plan.md` §5 for the concrete pattern.

Reconciliation, once both branches report back:
- If both agents' `outcome` is `answered` and `cohort_result.total_patients_matched <= 25` and both agents' drug counts match → high confidence, mark `mode="reconciled"`, use Role 1's citations (it saw the whole cohort).
- If both agents' `outcome` is `answered` and `cohort_result.total_patients_matched > 25` → Role 1's count is known-incomplete by definition. Use Role 2's exhaustive counts as authoritative; confidence tier recomputed off the *true* patient count; caveat states citations only cover the ≤25 patients RAG retrieved, not the full cohort.
- If both agents' `outcome` is `answered` but counts disagree despite `total_patients_matched <= 25` → do not silently pick one; set `confidence="low"`, `discrepancy_flag=True`, `authoritative_source="neither"`, surface both numbers for human review.
- If one agent's `outcome` is `nothing_found` and the other's is `answered` → this is a genuine disagreement, not a tool failure, and the two agents should not be filtering the same underlying population if this happens. Treat it the same as a count mismatch (`confidence="low"`, `discrepancy_flag=True`, `authoritative_source="neither"`), and additionally set a `notes` value on `ReconciliationResult` calling out that a `nothing_found`/`answered` split usually means Block 4's structured filter values (`condition`, `lab`) and Block 3's stored graph properties (`condition_name`, lab properties) don't share an identical vocabulary for this question, rather than a real absence of matching patients — needs human review before trusting either answer.
- If both agents' `outcome` is `nothing_found` → reconciled, `mode="reconciled"`, zero patients, no discrepancy.

**Answer-text synthesis when Role 1 is unavailable:** `cohort_only_degraded` mode has no LLM-authored sentence to fall back on — Role 2 never writes natural language, by design, and `CohortResult` has no `answer` field at all. But `MultiAgentAnswer.answer` is a required string in every mode, including this one. The reconcile node needs its own minimal answer-text generator for this specific case (e.g. a plain template built from `total_patients_matched`/`drug_a_count`/`drug_b_count`) — decide in plan.md whether that's a hardcoded template or a lightweight LLM call, but this field can't be left unpopulated when Role 1 has failed.

**Vocabulary consistency requirement:** Role 1's structured filters (passed to Block 4) and Role 2's graph properties (queried in Block 3) must refer to conditions and labs using the same strings — enforced by a startup/CI check cross-referencing Block 3's distinct `condition_name`/lab values against the vocabulary Block 4 expects (see `docs/plan.md` §12), so a count mismatch is treated as a real reconciliation-worthy disagreement rather than silent vocabulary drift.

**Confidence tiers are redefined for this system, not inherited unmodified from Block 5:** Block 5's tiers (`low` <15, `medium` 15–24, `high` ≥25) are calibrated around `top_k=25` being the maximum achievable patient count. Role 2 removes that ceiling, so reusing the same thresholds unmodified would let any common condition with a real cohort of, say, 40 patients trivially reach `high` — the tier would stop distinguishing anything once the ceiling it was calibrated against is gone. Block 6 instead ties confidence to *how* the count was obtained rather than *how big* it is — see `docs/plan.md` §8 for the tier definitions.

## 3. Explicit, Inspectable State

```python
class MultiAgentState(TypedDict):
    question: QuestionInput
    clinical_result: Optional[ClinicalAnswer]
    clinical_count_step_ran: Optional[bool]
    clinical_error: Optional[str]
    clinical_error_kind: Optional[Literal["timeout", "connection_error", "validation_error", "unknown"]]
    cohort_result: Optional[CohortResult]
    cohort_error: Optional[str]
    cohort_error_kind: Optional[Literal["timeout", "connection_error", "validation_error", "unknown"]]
    reconciliation: Optional[ReconciliationResult]
    final_answer: Optional[MultiAgentAnswer]
```
Every node writes its output into this dict rather than passing implicit context; the full state is dumped in the LangSmith trace at each transition, so any run's intermediate state is inspectable without re-running it. The `_error_kind` fields exist so degradation logic (and, later, Block 7's forensics) can distinguish an infrastructure failure from something worth flagging as suspicious, rather than treating every error identically. `clinical_count_step_ran` carries forward the second value of Block 5's `run_agent(...) -> tuple[ClinicalAnswer, bool]` — that boolean (`count_step_ran`) is otherwise silently dropped by this repo's clinical branch node. It matters for reconciliation: if `False`, Role 1 never reached its drug-counting step (e.g. it short-circuited on a fixed "nothing found" path), so its counts are not a real second opinion to compare Role 2 against — treat this the same as Role 1 having no comparable count at all, not as a `0` count that happens to disagree with Role 2.

New schemas live in this repo's own `scripts/schemas.py` — Block 5's `ClinicalAnswer` is composed into `MultiAgentAnswer`, never modified:
- `CohortResult`: `question: QuestionInput` (structured — Role 2 never touches RAG, so there's no reason to stringify it first), `total_patients_matched`, `drug_a_count`, `drug_b_count` (not required to sum to `total_patients_matched` — a patient can be on neither drug, one, or both, so this is not a partition and shouldn't be treated as a sanity-check invariant), `patient_ids` (full, unbounded), `outcome: Literal["answered","nothing_found","tool_error"]`, `caveat`.
- `ReconciliationResult`: `counts_match: bool`, `authoritative_source: Literal["clinical","cohort","neither"]` (`"neither"` covers the `nothing_found`/`answered` split and any unresolved count mismatch — see §2's reconciliation rules), `discrepancy_flag: bool`, `notes: str`.
- `Citation`: `patient_id: int`, `snippet: str`, `source: str` — one cited piece of evidence, not a bare string. `patient_id` and `snippet` map directly from Role 1's `ClinicalAnswer.rag_citations` entries (`{patient_id, chunk_id, snippet}` — see §2's Role 1 note); `source` is this repo's own label distinguishing where a citation came from, not a field Block 5 provides.
- `MultiAgentAnswer`: `question: str` (stringified, matching `ClinicalAnswer.question`'s existing convention — keep the final answer object consistent with Block 5 rather than introducing a second convention), `answer`, `total_patients`, `drug_a_count`, `drug_b_count`, `confidence: Literal["high","medium","low"]`, `mode: Literal["reconciled","clinical_only_degraded","cohort_only_degraded","both_failed"]`, `citations: list[Citation]`, `caveat`, `discrepancy_flag`.

## 4. Failure / Graceful Degradation Matrix

"Succeeds" below means `outcome` is `answered` *or* `nothing_found` — both are legitimate, non-error results the tool call itself completed normally. Only `tool_error` represents an actual failure (Neo4j down, RAG service down, timeout exhausted after retries). Genuine `answered`/`nothing_found` splits *between* the two agents (one found patients, the other found none) are a reconciliation case, not a degradation case — handled in §2's reconciliation rules, not this matrix.

| Clinical agent | Cohort agent | Behavior |
|---|---|---|
| succeeds | succeeds | Reconcile per §2. `mode="reconciled"` (or flagged per §2 if `answered`/`nothing_found` split). |
| `outcome="tool_error"` | succeeds | Use cohort agent's exhaustive count alone, no citations. `mode="cohort_only_degraded"`. Caveat: no supporting evidence available. |
| succeeds | `outcome="tool_error"` (Neo4j down/timeout) | Use clinical agent's answer as-is, with its existing top_k-ceiling caveat. `mode="clinical_only_degraded"`. |
| `outcome="tool_error"` | `outcome="tool_error"` | Return a fully-formed `MultiAgentAnswer` with `mode="both_failed"`, `confidence="low"`, explicit "unable to answer" text. Never raise. |

This is the graceful-degradation requirement from the assignment, made concrete and testable.

## 5. Dependencies

- Block 5 (`genai-block5-agent`): `run_agent`, `ClinicalAnswer`, `QuestionInput`. Signatures and field lists match this spec's description exactly — `run_agent(question, *, search_fn=search_patients, count_fn=count_drugs, answer_fn=_default_answer_fn) -> tuple[ClinicalAnswer, bool, dict]`, `top_k=25`, confidence tiers `low <15 / medium 15–24 / high ≥25`. `ClinicalAnswer.rag_citations` backs `Citation.snippet` — see Role 1's note above. Block 5's interface has changed several times over its own history — confidence tiers recalibrated twice, `outcome` added late, `top_k` raised from 20 to 25, a third `cost_info` return value added during Block 6's Phase 5 — so this contract is not assumed permanently frozen; re-confirm it if Block 5 is touched again in the future.
- Block 4 (`genai-block4-rag-eval`): called transitively via Block 5's `search_fn`, not directly by this repo. `SEARCH_TIMEOUT_SECONDS = 10` (`retrieve.py:69`) and `GENERATE_TIMEOUT_SECONDS = 10` (`generate.py:31`); both wrapped in one `except Exception` in `api.py`'s `/query` handler that returns a generic 502. `/query`'s response `sources` entries include `chunk_text` (`api.py:145`), which Block 5's `build_rag_citations()` depends on for `Citation.snippet`.
- Block 5's own outbound HTTP client to Block 4's `/query`: `rag_tool.py:60` sets `requests.post(..., timeout=10)` — tighter than Block 4's internal 10s+10s ceiling, not looser, since the client aborts at 10s regardless of what Block 4 might still be doing server-side. See `docs/plan.md` §7 for the worst-case math this produces.
- Block 3 (`genai-block3-graph-kb`): Neo4j instance/schema the Cohort Agent queries directly via Cypher. `HAS_CONDITION`/`PRESCRIBED` relationship names and `condition_name` property (`query_graph.py`); `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/`NEO4J_DATABASE` env-var convention (`check_connection.py`). Driver setup and relationship-name conventions are reusable (see §2).
- Note: Block 6 copies Block 5's seed file as-is (`docs/plan.md` §10), so the hand-written edge-case regression patient (person ID `900001`, `ci_graph_seed.cypher:102`) carries over automatically — no special handling needed. If Block 6's own seed data is ever regenerated or extended in the future, don't silently drop hand-written regression edge-case patients in the process — the same failure mode that briefly affected Block 5's seed rewrite.

## 6. Tracing & Evaluation (same bar as Block 5)

- Every run traced end-to-end (LangSmith or equivalent), tokens + latency captured per node, not just per run.
- Automated eval suite runs in CI: reuse Block 5's fixed question set — 11 questions total, 8 scored for recall (re-measuring the 2 previously-capped questions; target full recall now that the cohort agent is unbounded), plus 3 deliberately unanswerable control questions (Block 5's `answerable: false` entries) checked pass/fail that the system correctly reports no matches rather than scored for recall; add ≥1 new eval dimension exercising the failure matrix in §4 (inject synthetic tool failures via fakes, assert correct `mode` and no unhandled exception).
- Per-run cost + token usage logged.
- A regression in eval score (including a regression in the two previously-capped questions) fails the build.
- Spec written and committed first, per SDD convention; plan.md and tasks.md follow before any code.

**Cross-cutting concerns, resolved in `docs/plan.md`:**
- **PHI in traces.** §3's full state (patient IDs, RAG-cited note snippets) gets dumped into LangSmith at every node. Patient data across Blocks 3–6 is synthetic/de-identified, not real PHI, so this is accepted as-is with no redaction — see `docs/plan.md` §16.
- **Latency budget.** Role 1 alone can retry twice across two external calls (RAG, then its own LLM step) at up to 10s each; Role 2 adds its own retries on top. Expected and worst-case latency figures the eval harness checks against are set in `docs/plan.md` §7 and §13.
- **Cost of always running both agents.** Every question unconditionally costs Block 5's full agent run (LLM call included) plus a Neo4j round trip, even for questions where one agent's answer alone would have sufficed. The doubled cost is accepted as the point of the exercise — see `docs/plan.md` §17.
- **Observability export beyond LangSmith's own UI.** Block 8's capstone needs one dashboard showing traces, cost, and errors across the knowledge base, RAG service, and this multi-agent system together. Per-run cost/error data is also logged to a queryable file, not just traced — see `docs/plan.md` §9.
- **Eval reproducibility against a moving graph.** The Cohort Agent's enumeration query has no `LIMIT`, so its result depends on however much data is in Block 3's graph at query time. The eval harness runs against a frozen seed snapshot, not a live, mutable graph — see `docs/plan.md` §10.
- **Automated handling of `discrepancy_flag=True` in CI.** §2's reconciliation rules say to "surface for human review" when the two agents disagree — real production intent, not a CI behavior on its own, since the eval suite runs unattended with a pass/fail gate. Any `discrepancy_flag=True` result on the 8 fixed eval questions is treated as an eval failure, reported separately from the recall metric — see `docs/plan.md` §14.
- **The ground truth for the two previously-capped questions could itself be tainted by the same ceiling this block is fixing.** The claim in §7 is "recall improves on the 2 previously top_k-capped questions" — but if the golden answer key for those two questions was originally built by the same RAG-bounded process being replaced (i.e. the recorded "true" patient count was itself just whatever `top_k=25` could recall at the time, not an independently verified full count), then there's no correct number to measure Role 2's exhaustive count against, and the comparison would be circular. The true patient count for those two questions is independently re-verified — by running the same unbounded enumeration query once by hand against the seed data — rather than trusting the existing answer key as ground truth for exactly the case it was never able to measure correctly. See `docs/plan.md` §11 and `docs/tasks.md` Phase 4.
- **A real run produces two root LangSmith traces per question, not one nested trace.** Block 5's `run_agent` (Role 1) executes inside Block 6's dedicated thread-pool call from `clinical_node` — LangSmith's trace context doesn't propagate across that thread-pool boundary, so Role 1's own internal `StateGraph` (search → count → synthesize) is recorded as its own separate root trace, not nested under Block 6's outer orchestrator trace. Expected behavior, not a bug — correlate the two by timestamp (both start within a fraction of a second of the same triggering call) rather than expecting to find one under the other in the LangSmith UI.

## 7. Acceptance Criteria (done = all true)

- ≥2 agents with distinct roles (RAG-grounded Clinical QA Agent; unbounded structured Cohort Enumeration Agent) coordinate via a LangGraph orchestrator to complete a task neither does alone (exhaustive + cited cohort answer).
- Block 5's `run_agent` is reused as one role, never reimplemented or forked in this repo — its one interface change to date (adding `cost_info`, §2) was made in Block 5's own repo, per §8's rule that interface changes belong there, not worked around here.
- State is explicit (`MultiAgentState` TypedDict) and inspectable (full state visible in trace at every node transition).
- System degrades gracefully per the matrix in §4 when either agent fails — verified by eval tests with injected failures, never raises.
- Traced and evaluated as in Block 5: LangSmith tracing, CI eval suite, cost/token logging, regression gate.
- Cypher queries in the Cohort Agent are parameterized, not string-interpolated, and read-only (no `CREATE`/`MERGE`/`DELETE`/`SET`) — verified by code review / a targeted test.
- Measured recall on the two previously top_k-capped questions from Block 5's eval set improves (report before/after numbers — real measured, not assumed).

## 8. Out of Scope (deferred to later blocks)

- Threat modeling this system's new attack surface (Cypher injection, orchestrator state tampering) — Block 7.
- Deploying this system as part of the integrated capstone app — Block 8. Note for that work in advance: `run_multi_agent` (§2) is a plain Python callable here, not an HTTP service; Block 8 will need to wrap it in something like Block 4's FastAPI pattern, with its own env-based config for the Neo4j/Pinecone/Anthropic credentials this repo depends on. Not this block's job to build that wrapper, but the entry point above is named and typed with that future wrapping in mind.
- Modifying Block 5's or Block 4's interfaces — if either turns out to need a change, that change is proposed and made in *their* repos, not worked around here.

## 9. What I'd do next

- **Cost short-circuit** (deferred on purpose, `docs/plan.md` §17): both agents always run today, even on questions where one alone would suffice, because the assignment is specifically about two agents coordinating to complete a task neither does alone — short-circuiting would undercut that. A cheap pre-check (run the fast Cohort Agent first, only invoke Role 1's expensive LLM path when citations are actually needed for the final answer) is a real future optimization, explicitly not built here.
- **`block6_readonly` DB-level permission hardening.** Phase 4 found this isn't buildable at all on the current stack — Neo4j Community Edition has no role-based access control whatsoever (confirmed directly against a real instance of this project's own `neo4j:5.18-community` image: `SHOW ROLES` fails with `Unsupported administration command`, and `dbms.components()` reports `edition: "community"`). What's needed going forward is provisioning `block6_readonly` on Neo4j Enterprise Edition or an equivalent managed offering with RBAC support — not "harden further" on Community Edition, which has nothing further to harden. Application-level read-only Cypher (parameterized, `MATCH`/`RETURN`-only, §2) remains the sole enforced defense until then, not one layer of two (see `docs/plan.md` §6).
- **PHI-in-traces redaction**, if the synthetic-data assumption (`docs/plan.md` §16) ever stops holding. Full state — including patient IDs and RAG-cited note snippets — is dumped into LangSmith at every node transition, accepted as-is today only because patient data across Blocks 3–6 is synthetic/de-identified, not real PHI.
- **A real process-level driver shutdown hook** (`docs/plan.md` §6), needed once/if this repo is ever deployed as a long-running service (Block 8) rather than run as a short-lived CI/CLI process. Today, `scripts/cohort_tool.py::get_driver()`'s cached driver is only ever closed by a session-scoped test fixture at the end of a test run — there's no equivalent for a real deployment yet.
- ~~Verify the cost-per-token rate.~~ **Closed, verified current — not a remaining action.** `scripts/run_log.py`'s cost accounting inherits its $/token constants from Block 5's `logging_utils.py` (`$3`/`$15` per million input/output tokens for `claude-sonnet-4-6`), which documents itself as "not the source of truth for pricing." Checked directly against Anthropic's own published pricing docs (`platform.claude.com/docs/en/about-claude/pricing`): Claude Sonnet 4.6's standard API rate is listed as **$3 / MTok input, $15 / MTok output** — an exact match, no drift. Every `cost_usd` figure this system has reported to date (including in the README) is accurate at the rates in effect at verification time; re-check if Anthropic's pricing changes or this repo's model pin changes.
