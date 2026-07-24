# Block 6 — Multi-Agent Clinical Cohort System — Spec

Repo: `genai-block6-multiagent` (new, per portfolio strategy: one repo per block).

**A note on scope:** a few paragraphs below (the orchestrator's execution-model discussion, the Neo4j driver-lifecycle note, specific timeout mechanics) are properly plan-level "how to build it" detail, not spec-level "what and why." They're captured here because they surfaced during this spec's review, not because spec.md is meant to be their permanent home — carry them into plan.md when it's written, rather than leaving the same content duplicated in both documents.

## 1. Problem

Block 5 shipped a single clinical QA agent that answers "of patients with [condition] and [lab] [comparison] [value], how many are on [drug_a] vs [drug_b]?" It measured mean recall 0.789 across 8 questions, with 2 of 8 structurally capped: `count_drugs` only ever sees the patient IDs `search_patients` (RAG, `top_k` ≤ 25) handed it, so any cohort larger than 25 is undercounted by construction, not by retrieval failure. Block 5's spec explicitly deferred the fix: "for questions fully answerable via structured filters alone, the graph could enumerate every matching patient directly with no ceiling" — flagged as Block 6 candidate work.

Block 6 builds that fix as a second, distinct agent, and makes the *combination* of the two agents the actual deliverable: neither agent alone produces a cohort answer that is both **exhaustive** (correct count, no ceiling) and **evidence-grounded** (cited patient records). The Block 5 agent has citations but a capped count; the new agent has an exhaustive count but no citations. An orchestrator reconciles them.

## 2. Roles

**Role 1 — Clinical QA Agent (reused, unmodified).**
Block 5's `run_agent(question, *, search_fn, count_fn, answer_fn) -> tuple[ClinicalAnswer, bool]`, imported as-is from the Block 5 package. Not edited in this repo. Its value: RAG-grounded citations (`rag_patient_ids`, source snippets via `graph_result`), capped at `top_k=25`.

**Role 2 — Cohort Enumeration Agent (new).**
Queries Neo4j directly via parameterized Cypher on `condition`, `lab`, `comparison`, `value` to enumerate *every* matching patient — no `top_k` ceiling — then counts `drug_a` vs `drug_b` over the full cohort. No RAG call, no citations. Its value: a correct total when the cohort exceeds 25.

**Runs unconditionally for every question, not gated by question type.** Block 5's deferred note (§1) scoped this idea to "a specific subset of questions... fully answerable via structured filters alone" — implying not every question would qualify. But `QuestionInput` (`condition`, `lab`, `comparison`, `value`, `drug_a`, `drug_b`) is *always* fully structured in this system; there's no free-text-only question variant either block handles. So for Block 6, that "subset" is the entire set of questions it will ever receive, and Role 2 runs on all of them unconditionally. Stating this explicitly rather than leaving it implicit, because it stops being true the moment a future block (e.g. Block 8's broader capstone) introduces a question type Role 2 can't safely answer this way — that gating decision would need to be added then, not assumed already handled here.

Entry point (mirrors Block 5's DI pattern for testability with fakes):
```
run_cohort_agent(question: QuestionInput, *, graph_query_fn=query_full_cohort, count_fn=count_drugs_exhaustive) -> tuple[CohortResult, bool]
```
Never raises on tool failure — same retry/degrade contract as Block 5 (2 retries on the Cypher call, matching `_MAX_TOOL_RETRIES = 2` at `agent.py:33`).

**Starting point, don't write from scratch:** Block 5's `graph_tool.py:41-46` already has a `VERIFY_PATIENTS_QUERY_TEMPLATE` — `MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition {{condition_name: $condition}}) WHERE p.{lab_property} {op} $value AND p.person_id IN $person_ids RETURN collect(DISTINCT p.person_id)`. Dropping just the trailing `AND p.person_id IN $person_ids` clause and running the rest of that exact query against the full patient population *is* the unbounded enumeration query this role needs — confirmed via read-only audit, this is a one-clause edit of an existing, already-parameterized query, not new Cypher. Reuse the same relationship names it depends on (`HAS_CONDITION`, `PRESCRIBED` — also used identically in Block 3's `query_graph.py:21-29`) and the same driver/env-var convention (`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/`NEO4J_DATABASE`, matching `graph_tool.py:24-27` and Block 3's `check_connection.py:1-28`) so the Cohort Agent's Cypher stays consistent with both repos it sits between.

**Driver lifecycle constraint:** the Neo4j driver must be a long-lived, reused connection, not reconstructed per query — see `docs/plan.md` §6 for the concrete decision (a shared, injectable driver instance) and the reasoning behind it.

**Security constraints (flagging now for Block 7's threat model):**
- Cypher must be built with driver-level query parameters, never string interpolation of `condition`/`lab`/`drug_a`/`drug_b` values — these fields trace back to user-facing input once Block 8 integrates this into the capstone's app layer.
- The Cohort Agent's Cypher must be read-only (`MATCH`/`RETURN` only) — no `CREATE`/`MERGE`/`DELETE`/`SET`. This is a new code path with direct, unmediated access to a graph DB other blocks also depend on; it has no business need to write anything, so it shouldn't be able to.
- Application-code read-only (above) is one layer, not the whole defense: the Neo4j database user/role this driver connects as should also be scoped to read-only at the DB level, not just trusted to behave via query shape. Decide the DB user's actual permissions in plan.md rather than relying on Cypher-string discipline alone.
- Inherited, not introduced here: Role 1's LLM reads raw patient note text via RAG, and that text flows into `MultiAgentAnswer.answer` with no sanitization step. This is an indirect-prompt-injection surface (OWASP LLM Top 10) that predates Block 6, but Block 6 is the first place it feeds into a reconciled, multi-source answer — worth a paper trail here so Block 7 doesn't discover it cold.

**Orchestrator (LangGraph `StateGraph`).**
Public entry point — the actual "task neither agent does alone" deliverable Block 8 will import:
```
run_multi_agent(question: QuestionInput, *, clinical_agent_fn=run_agent, cohort_agent_fn=run_cohort_agent) -> MultiAgentAnswer
```
Runs both agents via LangGraph's native fan-out — two edges out of a single dispatch node, one to each agent, joined at a reconcile node — not hand-managed asyncio. Decided with mentor input: this gives a genuine latency win (bounded by the slower branch, not the sum of both) with the concurrency mechanics handled declaratively by the framework, and is the more instructive orchestration artifact for this block. Sequential execution remains an acceptable, simpler fallback if a first cut needs to ship fast — losing nothing conceptually — but should still route through LangGraph edges, not raw asyncio, either way.

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

**Assumption this reconciliation logic depends on, not yet verified:** Role 1's structured filters (passed to Block 4) and Role 2's graph properties (queried in Block 3) refer to conditions and labs using the same strings. Nothing in this repo enforces that today — worth a validation step (e.g., a startup check cross-referencing Block 3's distinct `condition_name`/lab values against whatever vocabulary Block 4 expects) before treating count mismatches as reconciliation-worthy "disagreements" rather than silent vocabulary drift.

**Confidence tiers need a recalibration decision, not silent reuse:** Block 5's tiers (`low` <15, `medium` 15–24, `high` ≥25) were calibrated specifically around `top_k=25` being the maximum achievable patient count. Once Role 2 removes that ceiling, reusing the same thresholds unmodified means any common condition with a real cohort of, say, 40 patients trivially reaches `high` — the tier stops distinguishing anything once the ceiling it was calibrated against is gone. Decide in plan.md whether Block 6 needs new thresholds (e.g. scaled to the actual distribution of cohort sizes in the seed data) rather than inheriting Block 5's numbers as-is.

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
- `Citation`: `patient_id: int`, `snippet: str`, `source: str` — one cited piece of evidence, not a bare string.
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

- Block 5 (`genai-block5-agent`): `run_agent`, `ClinicalAnswer`, `QuestionInput`. Re-verify these against the current state of the repo before implementation starts — Block 5's interface has changed more than once across its own history (confidence tiers recalibrated twice, `outcome` added late, `top_k` raised from 20 to 25), so don't assume this spec's description of them is still current by the time coding begins.
- Block 4 (`genai-block4-rag-eval`): called transitively via Block 5's `search_fn`, not directly by this repo. Both its external calls (Pinecone `search()`, Claude `messages.create()`) are configured with a 10s timeout, and both timeout errors fall through to a generic 502 response — confirm this is still accurate before relying on it in any degradation-timing decision.
- Block 3 (`genai-block3-graph-kb`): Neo4j instance/schema the Cohort Agent queries directly via Cypher; driver setup and relationship-name conventions confirmed reusable (see §2).
- Known inherited gap: Block 5's regenerated CI seed data doesn't yet include a hand-written edge-case regression patient (person ID `900001`) — tracked as an open, unchecked item in Block 5's `docs/tasks.md`, with a documented reason (a later seed-generation rewrite for shared-patient/combined-constraint handling never accounted for this edge case, so it needs real design work to port forward, not a mechanical reapply). Block 6's eval seed data should either include this patient or explicitly note the same gap — not silently drop it.

## 6. Tracing & Evaluation (same bar as Block 5)

- Every run traced end-to-end (LangSmith or equivalent), tokens + latency captured per node, not just per run.
- Automated eval suite runs in CI: reuse Block 5's 8-question fixed set, re-measure recall on the 2 previously-capped questions (target: full recall now that the cohort agent is unbounded); add ≥1 new eval dimension exercising the failure matrix in §4 (inject synthetic tool failures via fakes, assert correct `mode` and no unhandled exception).
- Per-run cost + token usage logged.
- A regression in eval score (including a regression in the two previously-capped questions) fails the build.
- Spec written and committed first, per SDD convention; plan.md and tasks.md follow before any code.

**Flagged, not yet resolved — decide in plan.md:**
- **PHI in traces.** §3's full state (patient IDs, RAG-cited note snippets) gets dumped into LangSmith at every node. That's clinical data going to a third-party tracing service with no redaction or scoping decided yet. At minimum, plan.md should state whether this is accepted as-is for a portfolio project's synthetic/de-identified data, or whether specific fields get excluded from the trace payload before Block 7 has to threat-model it properly.
- **Latency budget.** No stated worst-case time per question. Role 1 alone can retry twice across two external calls (RAG, then its own LLM step) at up to 10s each; Role 2 adds its own retries on top. Plan.md should set an expected/worst-case latency so the eval harness has something to check against, especially once parallel fan-out (§2) is in place.
- **Cost of always running both agents.** Every question now unconditionally costs Block 5's full agent run (LLM call included) plus a Neo4j round trip, even for questions where one agent's answer alone would have sufficed. Worth a documented decision in plan.md: accept the doubled cost as the point of the exercise, or note a future short-circuit as out of scope.
- **Observability export beyond LangSmith's own UI.** Block 8's capstone needs one dashboard showing traces, cost, and errors across the knowledge base, RAG service, and this multi-agent system together. If cost/error data only lives inside LangSmith's UI with no programmatic export, aggregating it into a cross-system dashboard later could be awkward. Decide now whether per-run cost/error data also gets logged somewhere queryable (a file, a table), not just traced.
- **Eval reproducibility against a moving graph.** The Cohort Agent's enumeration query has no `LIMIT`, so its result depends on however much data is in Block 3's graph at query time. If that graph's contents can change over time, the "true" cohort size for the eval's fixed questions could silently drift between runs. State whether the eval harness runs against the same frozen seed snapshot Block 5's CI already uses, or its own pinned snapshot — not a live, mutable graph.
- **Automated handling of `discrepancy_flag=True` in CI.** §2's reconciliation rules say to "surface for human review" when the two agents disagree — but the eval suite in this section runs unattended in CI with a pass/fail gate, and there's no human in that loop. Plan.md must define what the eval harness actually does when a fixed eval question produces a discrepancy: log it as a distinct metric separate from pass/fail, treat it as an eval failure requiring investigation, or accept it as a known, expected outcome for specific seeded questions. §2's "human review" language describes the intent for real production use, not a defined CI behavior — the two need to be reconciled explicitly, not left to whichever one an implementer notices first.
- **The ground truth for the two previously-capped questions may itself be tainted by the same ceiling this block is fixing.** The claim in §7 is "recall improves on the 2 previously top_k-capped questions" — but if the golden answer key for those two questions was originally built by the same RAG-bounded process being replaced (i.e. the recorded "true" patient count was itself just whatever `top_k=25` could recall at the time, not an independently verified full count), then there's no correct number to measure Role 2's exhaustive count against, and the comparison is circular. Before claiming recall improved, independently re-verify the true patient count for those two specific questions — e.g. by running the same unbounded enumeration query once by hand against the seed data — rather than trusting the existing answer key as ground truth for exactly the case it was never able to measure correctly.

## 7. Acceptance Criteria (done = all true)

- ≥2 agents with distinct roles (RAG-grounded Clinical QA Agent; unbounded structured Cohort Enumeration Agent) coordinate via a LangGraph orchestrator to complete a task neither does alone (exhaustive + cited cohort answer).
- Block 5's `run_agent` is reused unmodified as one role.
- State is explicit (`MultiAgentState` TypedDict) and inspectable (full state visible in trace at every node transition).
- System degrades gracefully per the matrix in §4 when either agent fails — verified by eval tests with injected failures, never raises.
- Traced and evaluated as in Block 5: LangSmith tracing, CI eval suite, cost/token logging, regression gate.
- Cypher queries in the Cohort Agent are parameterized, not string-interpolated, and read-only (no `CREATE`/`MERGE`/`DELETE`/`SET`) — verified by code review / a targeted test.
- Measured recall on the two previously top_k-capped questions from Block 5's eval set improves (report before/after numbers — real measured, not assumed).

## 8. Out of Scope (deferred to later blocks)

- Threat modeling this system's new attack surface (Cypher injection, orchestrator state tampering) — Block 7.
- Deploying this system as part of the integrated capstone app — Block 8. Note for that work in advance: `run_multi_agent` (§2) is a plain Python callable here, not an HTTP service; Block 8 will need to wrap it in something like Block 4's FastAPI pattern, with its own env-based config for the Neo4j/Pinecone/Anthropic credentials this repo depends on. Not this block's job to build that wrapper, but the entry point above is named and typed with that future wrapping in mind.
- Modifying Block 5's or Block 4's interfaces — if either turns out to need a change, that change is proposed and made in *their* repos, not worked around here.

## 9. Open Questions (resolve in plan.md, not here)

- The first cut uses parallel execution via LangGraph's native fan-out (see §2), not raw asyncio.
- Re-check Block 3/4/5's current interfaces against §5 before implementation starts — this spec's audit is a point-in-time snapshot (2026-07-24), not a guarantee those repos haven't changed since.
- Validate the condition/lab vocabulary assumption flagged in §2 before treating cross-agent count mismatches as reconciliation-worthy disagreements.
- PHI-in-traces, latency budget, and always-run-both-agents cost (flagged in §6) — none of these are decided yet; plan.md must take a position on each.
- Confidence-tier recalibration (flagged in §2) — decide whether Block 6 needs new thresholds rather than inheriting Block 5's `top_k=25`-calibrated tiers unmodified.
- The Neo4j database user's actual DB-level permissions (flagged in §2's security constraints) — decide the specific role/grant, not just the application-level read-only Cypher discipline.
- Explicit timeout values for the Cohort Agent's Cypher call and for each parallel branch at the graph level (flagged in §2) — neither has a stated number yet.
- Observability export beyond LangSmith, and eval reproducibility against a frozen graph snapshot (flagged in §6).
- How the eval harness scores a `discrepancy_flag=True` result in CI (flagged in §6) — "human review" isn't an automatable step, plan.md needs to define the actual pass/fail behavior.
- Confirm `clinical_count_step_ran=False` is correctly treated as "no comparable count" rather than a `0` count in the reconciliation logic (flagged in §3) once implemented.
- Whether branches are implemented as async nodes (`.ainvoke()`) or run via a thread pool for sync nodes (flagged in §2) — the parallel latency win doesn't happen automatically just from the graph's shape.
- Whether the Cohort Agent accepts/reuses a shared, long-lived Neo4j driver instance rather than constructing one per call (flagged in §2).
- Independently re-verify the true patient count for the two previously-capped eval questions before trusting "recall improved" (flagged in §6) — the existing answer key may share the same ceiling this block is fixing.
