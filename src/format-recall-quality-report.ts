import { RecallProjectIdentitySource } from './enums.js';
import type { LoadedRecallQualityCorpus, RecallQualityGate } from './recall-quality-corpus.js';
import type { RecallQualityEvaluationResult } from './run-recall-quality-evaluation.js';
import type {
  RecallQualityConfigurationMeasurement,
  RecallQualityGateCombination,
} from './select-recall-quality-policy.js';

/** Runtime, revision, and local-model identity required to reproduce a quality report. */
export interface RecallQualityReportEnvironment {
  command: string;
  gitCommit: string;
  gitDirty: boolean;
  nodeVersion: string;
  platform: string;
  architecture: string;
  cpuModel: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingServedModelId: string;
  embeddingArtifact: string;
  embeddingDimensions: number;
  rerankerBaseUrl: string;
  rerankerModel: string;
}

function escapeMarkdownTable(value: string): string {
  return value.replaceAll('|', '\\|').replaceAll('\n', ' ');
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMilliseconds(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function formatChunkPolicy(chunkPolicy: { maxTokens: number; overlapTokens: number }): string {
  return `${chunkPolicy.maxTokens}/${chunkPolicy.overlapTokens}`;
}

function formatExpectedSource(
  expectedSource: LoadedRecallQualityCorpus['specification']['cases'][number]['expectedSources'][number],
): string {
  const qualifiers = [
    expectedSource.expectedEvidenceKind,
    expectedSource.expectedSummaryKind,
    expectedSource.expectedBranch,
    expectedSource.expectedEvidenceRelation,
    `origin ${expectedSource.expectedSessionOrigin}`,
    `contributors ${expectedSource.requiredContributingEntryIds.join('+')}`,
  ].filter((value) => value !== undefined);
  const suffix = qualifiers.length > 0 ? ` (${qualifiers.join(', ')})` : '';
  return `${expectedSource.sessionFile}#${expectedSource.entryId}${suffix}`;
}

function formatGateThresholds(gate: RecallQualityGate): string[] {
  return [
    '| Metric | Frozen threshold |',
    '| --- | ---: |',
    `| Candidate-pool recall | ≥ ${formatRate(gate.minimumCandidatePoolRecall)} |`,
    `| Fused top-N recall | ≥ ${formatRate(gate.minimumFinalRecall)} |`,
    `| Context usefulness | ≥ ${formatRate(gate.minimumContextUsefulness)} |`,
    `| Source-occurrence preservation | ≥ ${formatRate(gate.minimumSourceOccurrencePreservation)} |`,
    `| Final duplicate-result rate | ≤ ${formatRate(gate.maximumFinalDuplicateRate)} |`,
    `| Query p95 | ≤ ${formatMilliseconds(gate.maximumQueryP95Milliseconds)} |`,
  ];
}

function formatQualityMatrix(combinations: readonly RecallQualityGateCombination[]): string[] {
  const lines = [
    '| Chunk | Candidates/channel | Final | Pool recall | Final recall | Pool duplicates | Final duplicates | Context | Sources | Provenance | Project p95 | Global p95 | Gate |',
    '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
  ];
  for (const combination of combinations) {
    lines.push(
      `| ${formatChunkPolicy(combination.chunkPolicy)} | ${combination.candidateCount} | ${combination.finalCount} | ${formatRate(combination.candidatePoolRecall)} | ${formatRate(combination.finalRecall)} | ${formatRate(combination.candidatePoolDuplicateRate)} | ${formatRate(combination.finalDuplicateRate)} | ${formatRate(combination.contextUsefulness)} | ${formatRate(combination.sourceOccurrencePreservation)} | ${formatRate(Math.min(combination.sessionOriginVerification, combination.evidenceRelationVerification, combination.contributingEntryVerification, combination.branchVerification))} | ${combination.queryLatencyByScope.project ? formatMilliseconds(combination.queryLatencyByScope.project.p95) : '—'} | ${combination.queryLatencyByScope.global ? formatMilliseconds(combination.queryLatencyByScope.global.p95) : '—'} | ${combination.gatePassed ? 'PASS' : 'FAIL'} |`,
    );
  }
  return lines;
}

function hasNoDiscriminatingRecallQualityVariance(
  combinations: readonly RecallQualityGateCombination[],
): boolean {
  if (combinations.length < 2) {
    return false;
  }
  const qualitySignatures = new Set(
    combinations.map((combination) =>
      [
        combination.candidatePoolRecall,
        combination.finalRecall,
        combination.contextUsefulness,
        combination.sourceOccurrencePreservation,
        combination.sessionOriginVerification,
        combination.evidenceRelationVerification,
        combination.contributingEntryVerification,
        combination.branchVerification,
        combination.finalDuplicateRate,
      ].join('|'),
    ),
  );
  return qualitySignatures.size === 1;
}

function findDecisionCombination(
  result: RecallQualityEvaluationResult,
): RecallQualityGateCombination | undefined {
  if (result.selection.selected) {
    return result.selection.selected;
  }
  return result.selection.combinations.toSorted(
    (left, right) =>
      left.failures.length - right.failures.length ||
      right.candidatePoolRecall - left.candidatePoolRecall ||
      right.finalRecall - left.finalRecall ||
      right.contextUsefulness - left.contextUsefulness ||
      left.candidateCount - right.candidateCount ||
      left.finalCount - right.finalCount,
  )[0];
}

function findDecisionConfiguration(
  result: RecallQualityEvaluationResult,
  decision?: RecallQualityGateCombination,
): RecallQualityConfigurationMeasurement | undefined {
  if (!decision) {
    return undefined;
  }
  return result.configurations.find(
    (configuration) =>
      configuration.chunkPolicy.id === decision.chunkPolicy.id &&
      configuration.candidateCount === decision.candidateCount,
  );
}

function formatCaseOutcomes(
  result: RecallQualityEvaluationResult,
  decision?: RecallQualityGateCombination,
): string[] {
  const configuration = findDecisionConfiguration(result, decision);
  if (!configuration || !decision) {
    return ['No measured configuration was available for per-case outcomes.'];
  }
  const lines = [
    `Shown for ${formatChunkPolicy(decision.chunkPolicy)}, ${decision.candidateCount} candidates/channel, and ${decision.finalCount} final results.`,
    '',
    '| Case | Scope | Boundary | Pool | Final | Context | Sources | Origin | Relation | Contributors | Branch | Raw/grouped | Query |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |',
  ];
  for (const caseMeasurement of configuration.measurement.caseMeasurements) {
    const finalMeasurement = caseMeasurement.finalCounts.find(
      ({ finalCount }) => finalCount === decision.finalCount,
    );
    if (!finalMeasurement) {
      continue;
    }
    lines.push(
      `| ${escapeMarkdownTable(caseMeasurement.caseId)} | ${caseMeasurement.scope} | ${caseMeasurement.searchScopeVerified && caseMeasurement.invocationProjectIdentityVerified && caseMeasurement.excludedSessionFilesAbsent && caseMeasurement.preLimitChannelsVerified ? 'pass' : 'fail'} | ${caseMeasurement.candidatePoolRecalled ? 'hit' : 'miss'} | ${finalMeasurement.finalRecalled ? 'hit' : 'miss'} | ${finalMeasurement.contextUseful ? 'useful' : 'fail'} | ${finalMeasurement.sourceOccurrencesPreserved ? `${finalMeasurement.preservedSourceOccurrences} kept` : `${finalMeasurement.preservedSourceOccurrences} fail`} | ${finalMeasurement.sessionOriginsVerified ? 'pass' : 'fail'} | ${finalMeasurement.evidenceRelationsVerified ? 'pass' : 'fail'} | ${finalMeasurement.contributingEntriesVerified ? 'pass' : 'fail'} | ${finalMeasurement.branchesVerified ? 'pass' : 'fail'} | ${caseMeasurement.rawCandidateCount}/${caseMeasurement.groupedCandidateCount} | ${formatMilliseconds(caseMeasurement.queryLatencyMilliseconds)} |`,
    );
  }
  if (configuration.measurement.caseMeasurements.length === 0) {
    lines.push(
      '| _No per-case rows in this fixture_ | — | — | — | — | — | — | — | — | — | — | — | — |',
    );
  }
  return lines;
}

function formatPreLimitChannelProofs(
  result: RecallQualityEvaluationResult,
  decision?: RecallQualityGateCombination,
): string[] {
  const configuration = findDecisionConfiguration(result, decision);
  const proofs =
    configuration?.measurement.caseMeasurements.flatMap((measurement) =>
      measurement.preLimitChannelMeasurements.map((channel) => ({
        caseId: measurement.caseId,
        ...channel,
      })),
    ) ?? [];
  if (proofs.length === 0) {
    return ['No pre-limit channel proof was declared.'];
  }
  return [
    '| Case | Channel | Project source admitted | Global source displaced | Polluters inside limit | Proof |',
    '| --- | --- | --- | --- | ---: | --- |',
    ...proofs.map(
      (proof) =>
        `| ${proof.caseId} | ${proof.channel} | ${proof.projectSourceAdmitted ? 'yes' : 'no'} | ${proof.globalSourceDisplaced ? 'yes' : 'no'} | ${proof.pollutingCandidateCount} | ${proof.passed ? 'PASS' : 'FAIL'} |`,
    ),
  ];
}

/** Formats one reproducible, reviewable project-scoped recall quality decision report. */
export function formatRecallQualityReport(
  result: RecallQualityEvaluationResult,
  corpus: LoadedRecallQualityCorpus,
  environment: RecallQualityReportEnvironment,
): string {
  const { specification } = corpus;
  const decision = findDecisionCombination(result);
  const lines: string[] = [
    '# Project-scoped recall quality evaluation before backfill',
    '',
    `Generated ${result.completedAt} from corpus \`${result.corpusId}\`.`,
    '',
    '## Decision',
    '',
    `**Automated gate: ${result.selection.passed ? 'PASS' : 'FAIL'}**`,
    '',
  ];
  if (result.selection.selected) {
    const selected = result.selection.selected;
    lines.push(
      `Selected **${formatChunkPolicy(selected.chunkPolicy)} tokens/overlap**, **${selected.candidateCount} candidates/channel**, and **${selected.finalCount} final results**. This is the smallest measured candidate count, then the smallest final count, that passes every frozen hybrid-search gate; p95 query latency breaks ties.`,
    );
  } else {
    lines.push('No candidate or final-result count passed every frozen gate.');
    for (const blocker of result.selection.blockers) {
      lines.push(`- ${blocker}`);
    }
  }
  lines.push(
    '',
    '**Full corpus backfill remains blocked pending human approval.** The command evaluated only the committed bounded corpus and did not read the configured production sessions directory.',
    '',
    '## Evaluation identity',
    '',
    `- Default scope: \`${result.evaluationIdentity.defaultScope}\` (policy v${result.evaluationIdentity.projectScopePolicyVersion})`,
    `- Project identity policy: v${result.evaluationIdentity.projectIdentityPolicyVersion}; metadata schema v${result.evaluationIdentity.projectIdentityMetadataSchemaVersion}`,
    `- Project lineage policy: v${result.evaluationIdentity.lineagePolicyVersion}; digest \`${result.evaluationIdentity.lineageDigest}\``,
    `- Hybrid ranking: fusion v${result.evaluationIdentity.rankFusionVersion}, RRF k=${result.evaluationIdentity.reciprocalRankConstant}, active prior +${result.evaluationIdentity.activeBranchPrior.toFixed(4)}`,
    `- Candidate limits: dense ${result.evaluationIdentity.candidateLimits.dense}, lexical ${result.evaluationIdentity.candidateLimits.lexical}, identifier ${result.evaluationIdentity.candidateLimits.identifier}; final results ${result.evaluationIdentity.finalResultCount}`,
    '',
    '## Frozen quality gate',
    '',
    ...formatGateThresholds(specification.qualityGate),
    '',
    'The gate and count grid live in `evaluation/recall-quality-cases.json`; the run does not alter them after seeing results.',
    '',
    '## Bounded work',
    '',
    '| Work | Executed | Hard maximum |',
    '| --- | ---: | ---: |',
    `| Session files/index | ${result.boundedWork.sessionFiles} | ${specification.bounds.maximumSessionFiles} |`,
    `| Evaluation cases | ${result.boundedWork.evaluationCases} | ${specification.bounds.maximumEvaluationCases} |`,
    `| Temporary index runs | ${result.boundedWork.indexRuns} | ${specification.bounds.maximumChunkPolicies} |`,
    `| Search requests, including warmups | ${result.boundedWork.executedSearchRequests} | ${specification.bounds.maximumSearchRequests} |`,
    `| Reranker requests | ${result.boundedWork.rerankerRequests} | 0 |`,
    `| Chunk-embedding HTTP batches | ${result.boundedWork.chunkEmbeddingRequests} | ${specification.bounds.maximumChunkEmbeddingRequests} |`,
    `| Maximum fused candidates/search | ${result.boundedWork.maximumCandidatesPerSearch} | 200 |`,
    `| Production repository identity resolutions | ${result.boundedWork.repositoryIdentityResolutions} | ${
      specification.projectIdentityFixtures.filter(
        ({ identitySource }) =>
          identitySource !== RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
      ).length
    } |`,
    '',
    `Run duration: ${formatMilliseconds(result.durationMilliseconds)}. Work data stayed under \`evaluation/.recall-data/${RECALL_QUALITY_WORK_DIRECTORY_NAME_FOR_REPORT}/\` and used only ${result.boundedWork.sessionFiles} checksum-fixed JSONL files.`,
    '',
    '## Metric definitions',
    '',
    '- **Candidate-pool recall:** fraction of cases whose declared source appears anywhere in the complete bounded fused pool before duplicate grouping.',
    '- **Fused top-N recall:** fraction of cases whose declared source appears in the first _N_ deterministic hybrid result groups.',
    '- **Duplicate-result rate:** slots duplicating an earlier exact cross-session copy or overlapping source span, divided by all slots. Candidate-pool measurement reconstructs raw candidates; final measurement uses visible result groups.',
    '- **Context usefulness:** fraction of cases whose first _N_ matching displayed results contain every independently declared context fragment. Neighbor-expanded text is used when present.',
    '- **Source-occurrence preservation:** fraction of cases retaining the required count of distinct declared source locations, including suppressed duplicate occurrences.',
    `- **Query latency:** wall time for the full read-only hybrid service search, measured and gated independently for project and global scope. Tables report nearest-rank p95 across ${specification.cases.length} fixed cases after ${specification.warmupQueriesPerCombination} warmup request per represented scope and configuration.`,
    '',
    '## Chunk-policy index comparison',
    '',
    '| Chunk | Stored documents | Scanned/indexed sessions | New embedded documents | Embedding batches | Index time |',
    '| ---: | ---: | ---: | ---: | ---: | ---: |',
  );
  for (const indexRun of result.indexRuns) {
    lines.push(
      `| ${formatChunkPolicy(indexRun.chunkPolicy)} | ${indexRun.totalChunks} | ${indexRun.indexSummary.scannedSessions}/${indexRun.indexSummary.indexedSessions} | ${indexRun.indexSummary.newlyEmbeddedChunks} | ${indexRun.indexSummary.embeddingRequestCount} | ${formatMilliseconds(indexRun.indexLatencyMilliseconds)} |`,
    );
  }
  lines.push(
    '',
    '## Quality and latency matrix',
    '',
    ...formatQualityMatrix(result.selection.combinations),
    '',
    '## Pre-limit channel proof',
    '',
    ...formatPreLimitChannelProofs(result, decision),
    '',
    '## Fixed cases and independent source evidence',
    '',
    '| Case | Category | Scope | Invocation | Query | Expected source evidence | Excluded sessions | Required context |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |',
  );
  for (const evaluationCase of specification.cases) {
    lines.push(
      `| ${evaluationCase.id} | ${evaluationCase.category} | ${evaluationCase.scope} | ${escapeMarkdownTable(evaluationCase.invocationDirectory ?? 'none')} | ${escapeMarkdownTable(evaluationCase.query)} | ${escapeMarkdownTable(evaluationCase.expectedSources.map(formatExpectedSource).join('<br>'))} | ${escapeMarkdownTable(evaluationCase.excludedSessionFiles.join('; ') || 'none')} | ${escapeMarkdownTable(evaluationCase.requiredContext.join('; '))} |`,
    );
  }
  lines.push(
    '',
    '## Per-case outcome',
    '',
    ...formatCaseOutcomes(result, decision),
    '',
    '## Reproduce',
    '',
    'Prerequisites: the pinned Octen tokenizer assets and configured local embedding endpoint must be available. The optional reranker is not called. The command deletes and recreates only the dedicated ignored evaluation work directory.',
    '',
    '```bash',
    environment.command,
    '```',
    '',
    'The command rewrites:',
    '',
    '- `docs/evaluation/recall-quality-report.md`',
    '- `docs/evaluation/recall-quality-results.json`',
    '',
    'Environment:',
    '',
    `- Git commit: \`${environment.gitCommit}\`${environment.gitDirty ? ' (dirty)' : ''}`,
    `- Node: \`${environment.nodeVersion}\``,
    `- Platform: \`${environment.platform}/${environment.architecture}\``,
    `- CPU: ${environment.cpuModel}`,
    `- Embedding: \`${environment.embeddingModel}\` → \`${environment.embeddingServedModelId}\`, \`${environment.embeddingArtifact}\`, ${environment.embeddingDimensions} dimensions at \`${environment.embeddingBaseUrl}\``,
    `- Hybrid ranking identity: fusion v${result.evaluationIdentity.rankFusionVersion}, RRF k=${result.evaluationIdentity.reciprocalRankConstant}, active prior +${result.evaluationIdentity.activeBranchPrior.toFixed(4)}`,
    `- Optional deep reranker, not used by this evaluation: \`${environment.rerankerModel}\` at \`${environment.rerankerBaseUrl}\``,
    `- Specification: \`${corpus.specificationPath}\``,
    `- Specification SHA-256: \`${corpus.specificationSha256}\``,
    '',
    'Corpus file checksums:',
    '',
  );
  for (const sessionFile of corpus.sessionFiles) {
    lines.push(`- \`${sessionFile.fileName}\`: \`${sessionFile.sha256}\``);
  }
  lines.push(
    '',
    '## Limits of this evidence',
    '',
    '- The corpus is a committed synthetic-but-session-shaped fixture, not a sample of private production logs. It covers the required retrieval and project-identity classes but cannot estimate all real-corpus failure modes.',
    ...(hasNoDiscriminatingRecallQualityVariance(result.selection.combinations)
      ? [
          '- The measured grid has no discriminating quality variance across gated recall, context, source-preservation, and visible-duplicate metrics; it can identify the smallest passing candidate pool but cannot rank quality among passing pools.',
        ]
      : []),
    '- Latency uses one measured request per case after one warmup, so it compares configurations on this host rather than establishing a capacity benchmark.',
    '- A passing automated gate supports a candidate policy; it does not authorize the full corpus backfill. Human review of this report remains the approval boundary.',
    '',
  );
  return `${lines.join('\n')}\n`;
}

const RECALL_QUALITY_WORK_DIRECTORY_NAME_FOR_REPORT = 'recall-quality-evaluation';
