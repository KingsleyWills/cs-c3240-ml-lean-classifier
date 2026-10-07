# Project review instructions

## Purpose and scope

Review trustmebro as a next-tactic-family prediction project whose deliverable is a course-compliant experiment and report.
Evaluate whether supporting machinery advances reliable data preparation, useful feature construction, fair model comparison, and defensible conclusions.
Do not judge success primarily by visualization quality, infrastructure sophistication, or code rearrangement.

By default, cover the active Stage 2 system: extraction and storage, labels and representations, features and vocabularies, training and evaluation, analysis and visualization, CLI/environment configuration, tests, and relevant documentation/report content.
A task may narrow this scope; follow its boundaries while inspecting interfaces and consumers needed to establish the relevant contracts.
Distinguish implemented behavior from proposed work and historical Stage 1 material.
Do not report a deliberately unimplemented future component as a code defect; identify a missing deliverable when it matters to the assigned readiness review.

Read `AGENTS.md` and the task's requirements before reviewing.
Consult `doc/internal/course-project-instructions.md`, `doc/internal/allowed-ml-methods.md`, and the supplied outline/grading criteria when assessing methods or submission readiness.
Use current user-approved decisions rather than treating an old TODO, label proposal, or implementation log as an immutable specification.

Write review deliverables only; use narrowly scoped temporary diagnostics where permitted.
Keep earlier assignments, findings, and implementation records unchanged so subsequent comparisons remain honest.

## Review dimensions and criteria

Apply these dimensions at system, file, type/class, function, and variable levels where relevant.

| Dimension | Questions |
| --- | --- |
| Project fit | Does this work advance the predictor, experiment, or required report? Is its scope proportionate to that need? |
| Responsibilities and boundaries | Are ownership, interfaces, and dependency directions apparent? Is work misplaced, duplicated, or mixed? |
| Delegation and library use | Should substantial work be elsewhere or delegated to established functionality? Do helpers express useful operations rather than fragment implementation? |
| Naming | Do names distinguish concepts, units, populations, identities, and stages consistently, without hiding meaningful differences? |
| Data flow and representation | Are inputs, transformations, mutation, precision, and outputs explicit? Do intermediate representations and conversions earn their cost? |
| Control flow and failures | Are assumptions, side effects, exceptions, cancellation, recovery, and artifact lifetimes understandable? |
| Experimental validity | Do labels, feature construction, data splitting, model selection, and evaluation support the claimed prediction task? |
| Verification and reporting | Do tests establish the intended behavior? Are conclusions traceable to actual evidence and appropriately limited? |

For significant findings, judge coherence, necessity, complexity, consistency, understandability, correctness, resource cost, and implementation leverage.
Explain what becomes worse if a component is removed and what a proposed alternative must preserve.
Prefer recurring consequential patterns over exhaustive local style complaints.
Minimize maintenance burden, not lines or file count alone; acknowledge both justified growth and unsuccessful simplification.

## Architectural and data-flow audit

Map the actual flow from corpus selection through extraction, labeling, feature construction, training, prediction, evaluation, and reported results.
Include supporting analysis products where they inform these stages.
Record entry points, representations, persistent artifacts, configuration, ownership, and failure paths.
Unavailable stages should be marked as planned, not described as inspected implementations.

Use responsibility boundaries as hypotheses, not prescribed filenames:

- Orchestration owns scheduling, source access, run configuration, and artifact lifecycles.
- Extraction owns observation of Lean execution and faithful export of relevant states/actions.
- Processing owns representation and label transformations, vocabulary construction, and feature encoding.
- Training owns fitted preprocessing and model state; evaluation owns comparison protocols and result calculation.
- Analysis owns measurements and aggregation; rendering owns images and display preparation.
- Reporting communicates methods, evidence, limitations, and course-required disclosures.

Check actual dependencies and interfaces rather than whether modules have matching names.
Shared preparation, global aggregation, and multiple passes are legitimate when required; repeated identical expensive work needs justification.
Keep exploratory products distinguishable from model inputs and final evaluation results.

Trace representative values from their origin through calculation, aggregation, serialization, and consumers.
Identify changes to units, precision, ordering, multiplicity, identity, population coverage, or recoverability.
For every material lossy transformation, establish its purpose, consumer, and approval: rounding, clipping, binning, logarithms, normalization, sampling, and graph/label simplification are not automatically harmless.
Verify that exact raw counts survive durable boundaries and that derived analytical products do not silently replace them.
Inspect unused fields, speculative descriptors, duplicate representations, and stored plot coordinates rather than accepting them because their names sound analytical.

## Domain and experimental checks

Select checks relevant to the assigned scope; do not turn every review into a full re-extraction or training run.

- **Extraction and labels:** Distinguish source syntax, macro expansion, successful dynamic executions, pre-action states, labels, and callable tactics.
  Check that transformations preserve the intended observations, including nested/repeated actions, rollback, and local/metavariable state where applicable.
  State omissions and coverage limits; do not conflate source occurrence counts with executed transitions.
- **Graph and feature semantics:** Check operand order, repeated references, DAG sharing, binding relationships, and variable identities where relevant.
  Explain what information the representation retains, merges, or discards, and whether the selected classifier can use it.
  Label prediction is not automatically executable-tactic generation or proof completion.
- **Splits and fitted state:** Keep theorem transitions together and inspect relevant duplication/dependency risks under the chosen generalization claim.
  Trace corpus-derived vocabulary selection and fitted preprocessing to training data; distinguish exploratory whole-corpus analysis from leakage-safe evaluation.
  Check that validation guides selection and test results are not repeatedly used for tuning.
- **Comparison and conclusions:** Check allowed-method eligibility, multidimensional inputs, comparable splits/protocols, justified baselines and metrics, class imbalance, and reproducible settings.
  Distinguish matching the source action from finding a valid alternative; do not infer tactic validity or proof-solving performance from classification accuracy alone.
  Verify that the report's method descriptions and numerical claims match the implementation and recorded experiments.
- **Course deliverables:** Where in scope, check the two-method/different-row requirement, comparison and selection, required code, report constraints, anonymity, and AI disclosure against the supplied instructions.
  Do not invent course requirements or silently treat provisional Stage 1 choices as mandatory Stage 2 decisions.

Respect the supported operating contract in `AGENTS.md`, including an immutable corpus during analysis and no concurrent extraction/visualization workloads.
Review whether these assumptions are stated and consistently relied upon; do not present unsupported concurrent mutation as a normal-workflow defect.
If an assumption is insufficient for the stated contract, explain the gap rather than silently extending the contract or dismissing the evidence.

## Library and resource investigation

For substantial, complex, or computationally expensive custom operations, investigate both existing dependencies/standard-library APIs and well-established external alternatives.
Actively search where the work corresponds to established functionality; absence from current dependencies is not proof that custom code is necessary.
Prefer compatibility with the project's Lean, NumPy, graph_tool, scikit-learn, MessagePack/Zstd, and plotting representations where relevant.

Assess semantic fit, maturity, documentation, integration effort, conversion/copying costs, expected runtime and peak memory, installation requirements, and dependency burden.
Use official documentation and primary sources to substantiate recommendations.
A mature library reduces the need to validate its internal algorithms, not the need to verify our semantic fit and integration.
Recommend additional dependencies only for a concrete benefit; do not install them during review.
Explain why domain-specific custom code should remain when alternatives cannot preserve its contract.

Follow expensive work through its full lifecycle: preparation, hashing/traversal, serialization, repeated passes, worker transport, retention, and cleanup.
Identify which populations and allocations grow with corpus size, expression depth, or feature dimensionality.
Distinguish bounded output batches, encoded frames, queues, indexes, and total process-tree memory.
Do not infer a global memory bound from a spill threshold or native implementation.
Performance claims require measurements or an identified mechanism; label inferred costs and unmeasured consequences explicitly.
Do not use samples to establish exact population behavior or small fixtures to claim full-corpus capacity.

## Procedure and verification

1. Establish the task's outcome, scope, operating assumptions, preservation requirements, exclusions, and authorized checks.
2. Record the reviewed revision and relevant working-tree state; inspect architecture before individual implementations.
3. Trace representative data and experimental decisions across boundaries, checking the contracts above.
4. Research library alternatives and identify recurring correctness, resource, and clarity problems.
5. Inspect tests at public boundaries and run only permitted, narrowly scoped checks needed to resolve consequential uncertainty.
6. Compare evidence against requirements and, for a follow-up review, against the original findings, implementation plan/log, and available before/after code.

Prefer end-to-end evidence that asserts values, identities, populations, fitted-state boundaries, results, and failure behavior, not merely output existence.
Explain when narrower tests are necessary and which failure cases they cover.
Mocked boundaries do not establish behavior of the components they replace; inspect whether a test exercises the ownership or semantic boundary claimed.
Historical passing results are not fresh verification, and deferred checks are neither passes nor failures.
Agree input, worker count, resource budget, and stop criteria before full-corpus workloads or resource-intensive benchmarks.

## Deliverables and assessment

Write new results under a distinct subdirectory of `doc/internal/review`, named for the assigned review.
Do not overwrite historical findings or implementation evidence unless explicitly asked.
Keep deliverables proportionate to scope; they may share files when separate documents would only duplicate information.

Include:

- A compact architecture/data-flow map and proposed interface changes where warranted.
- Prioritized findings with code/artifact references, applicability, evidence, consequences, alternatives, and trade-offs.
- Evidence-backed library recommendations and components/custom semantics worth retaining.
- A scoped follow-up plan separating independent and coordinated changes, with behavioral acceptance criteria and verification limits.
- A record of checks actually executed, their results, and important unverified assumptions.

For follow-up reviews, map each material original requirement/finding to current implementation, evidence, remaining gaps, and status.
Distinguish achieved, partially achieved, implemented-but-unverified, and explicitly deferred/excluded outcomes.
Assess whether reasons for remaining work are sufficient; an implementation limitation does not silently amend acceptance criteria.
Do not attribute all cumulative changes to one author or claim unavailable intermediate states were inspected.
Separate demonstrated defects, supported-workflow limitations, unsupported-workflow hazards, resource mechanisms, and stylistic preferences.

Prioritize course/experimental validity, correctness and safety, and consequential resource risks before architecture and local readability.
Avoid arbitrary numerical grades, exhaustive minor nits, and claims that all goals are met merely because planned edits were made.
