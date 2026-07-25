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
  ].filter((value) => value !== undefined);
  const suffix = qualifiers.length > 0 ? ` (${qualifiers.join(', ')})` : '';
  return `${expectedSource.sessionFile}#${expectedSource.entryId}${suffix}`;
}

function formatGateThresholds(gate: RecallQualityGate): string[] {
  return [
    '| Metric | Frozen threshold |',
    '| --- | ---: |',
    `| Pre-rerank recall | ≥ ${formatRate(gate.minimumPreRerankRecall)} |`,
    `| Post-rerank recall | ≥ ${formatRate(gate.minimumPostRerankRecall)} |`,
    `| Context usefulness | ≥ ${formatRate(gate.minimumContextUsefulness)} |`,
    `| Source-occurrence preservation | ≥ ${formatRate(gate.minimumSourceOccurrencePreservation)} |`,
    `| Post-rerank duplicate-result rate | ≤ ${formatRate(gate.maximumPostRerankDuplicateRate)} |`,
    `| Query p95 | ≤ ${formatMilliseconds(gate.maximumQueryP95Milliseconds)} |`,
    `| Reranker p95 | ≤ ${formatMilliseconds(gate.maximumRerankerP95Milliseconds)} |`,
  ];
}

function formatQualityMatrix(combinations: readonly RecallQualityGateCombination[]): string[] {
  const lines = [
    '| Chunk | Candidates/channel | Final | Pre recall | Post recall | Pre duplicates | Post duplicates | Context | Sources | Query p50/p95 | Reranker p50/p95 | Gate |',
    '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
  ];
  for (const combination of combinations) {
    lines.push(
      `| ${formatChunkPolicy(combination.chunkPolicy)} | ${combination.candidateCount} | ${combination.finalCount} | ${formatRate(combination.preRerankRecall)} | ${formatRate(combination.postRerankRecall)} | ${formatRate(combination.preRerankDuplicateRate)} | ${formatRate(combination.postRerankDuplicateRate)} | ${formatRate(combination.contextUsefulness)} | ${formatRate(combination.sourceOccurrencePreservation)} | ${formatMilliseconds(combination.queryLatencyMilliseconds.median)} / ${formatMilliseconds(combination.queryLatencyMilliseconds.p95)} | ${formatMilliseconds(combination.rerankerLatencyMilliseconds.median)} / ${formatMilliseconds(combination.rerankerLatencyMilliseconds.p95)} | ${combination.gatePassed ? 'PASS' : 'FAIL'} |`,
    );
  }
  return lines;
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
      right.preRerankRecall - left.preRerankRecall ||
      right.postRerankRecall - left.postRerankRecall ||
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
    '| Case | Category | Pre | Post | Context | Sources | Raw/grouped | Query | Reranker |',
    '| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |',
  ];
  for (const caseMeasurement of configuration.measurement.caseMeasurements) {
    const finalMeasurement = caseMeasurement.finalCounts.find(
      ({ finalCount }) => finalCount === decision.finalCount,
    );
    if (!finalMeasurement) {
      continue;
    }
    lines.push(
      `| ${escapeMarkdownTable(caseMeasurement.caseId)} | ${caseMeasurement.category} | ${caseMeasurement.preRerankRecalled ? 'hit' : 'miss'} | ${finalMeasurement.postRerankRecalled ? 'hit' : 'miss'} | ${finalMeasurement.contextUseful ? 'useful' : 'fail'} | ${finalMeasurement.sourceOccurrencesPreserved ? `${finalMeasurement.preservedSourceOccurrences} kept` : `${finalMeasurement.preservedSourceOccurrences} fail`} | ${caseMeasurement.rawCandidateCount}/${caseMeasurement.groupedCandidateCount} | ${formatMilliseconds(caseMeasurement.queryLatencyMilliseconds)} | ${formatMilliseconds(caseMeasurement.rerankerLatencyMilliseconds)} |`,
    );
  }
  if (configuration.measurement.caseMeasurements.length === 0) {
    lines.push('| _No per-case rows in this fixture_ | — | — | — | — | — | — | — | — |');
  }
  return lines;
}

/** Formats one reproducible, reviewable issue-8 decision report from raw measurements. */
export function formatRecallQualityReport(
  result: RecallQualityEvaluationResult,
  corpus: LoadedRecallQualityCorpus,
  environment: RecallQualityReportEnvironment,
): string {
  const { specification } = corpus;
  const decision = findDecisionCombination(result);
  const lines: string[] = [
    '# Recall quality evaluation before backfill',
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
      `Selected **${formatChunkPolicy(selected.chunkPolicy)} tokens/overlap**, **${selected.candidateCount} candidates/channel**, and **${selected.finalCount} final results**. This is the smallest measured candidate count, then the smallest final count, that passes every frozen gate; p95 query and reranker latency break ties.`,
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
    `| Reranker requests | ${result.boundedWork.rerankerRequests} | ${specification.bounds.maximumSearchRequests} |`,
    `| Chunk-embedding HTTP batches | ${result.boundedWork.chunkEmbeddingRequests} | bounded by ${result.boundedWork.sessionFiles} files/index |`,
    `| Maximum fused candidates/search | ${result.boundedWork.maximumCandidatesPerSearch} | 200 |`,
    '',
    `Run duration: ${formatMilliseconds(result.durationMilliseconds)}. Work data stayed under \`.recall-data/${RECALL_QUALITY_WORK_DIRECTORY_NAME_FOR_REPORT}/\` and used only ${result.boundedWork.sessionFiles} checksum-fixed JSONL files.`,
    '',
    '## Metric definitions',
    '',
    '- **Pre-rerank recall:** fraction of cases whose declared source appears anywhere in the complete bounded fused pool before duplicate grouping and reranking.',
    '- **Post-rerank recall:** fraction of cases whose declared source appears in the first _N_ reranked result groups.',
    '- **Duplicate-result rate:** slots duplicating an earlier exact cross-session copy or overlapping source span, divided by all slots. Pre-rerank uses reconstructed raw candidates; post-rerank uses visible result groups.',
    '- **Context usefulness:** fraction of cases whose first _N_ matching displayed results contain every independently declared context fragment. Neighbor-expanded text is used when present.',
    '- **Source-occurrence preservation:** fraction of cases retaining the required count of distinct declared source locations, including suppressed duplicate occurrences.',
    `- **Query latency:** wall time for the full read-only service search. **Reranker latency:** wall time inside the local reranker request. Tables report nearest-rank median and p95 across ${specification.cases.length} fixed cases after ${specification.warmupQueriesPerCombination} warmup request per configuration.`,
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
    '## Fixed cases and independent source evidence',
    '',
    '| Case | Category | Query | Expected source evidence | Required context |',
    '| --- | --- | --- | --- | --- |',
  );
  for (const evaluationCase of specification.cases) {
    lines.push(
      `| ${evaluationCase.id} | ${evaluationCase.category} | ${escapeMarkdownTable(evaluationCase.query)} | ${escapeMarkdownTable(evaluationCase.expectedSources.map(formatExpectedSource).join('<br>'))} | ${escapeMarkdownTable(evaluationCase.requiredContext.join('; '))} |`,
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
    'Prerequisites: the pinned Octen tokenizer assets and the configured local embedding and reranker endpoints must be available. The command deletes and recreates only the dedicated ignored evaluation work directory.',
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
    `- Reranker: \`${environment.rerankerModel}\` at \`${environment.rerankerBaseUrl}\``,
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
    '- The corpus is a committed synthetic-but-session-shaped fixture, not a sample of private production logs. It covers every required retrieval class and includes 48 distractors plus a long boundary case, but it cannot estimate all real-corpus failure modes.',
    '- Latency uses one measured request per case after one warmup, so it compares configurations on this host rather than establishing a capacity benchmark.',
    '- A passing automated gate supports a candidate policy; it does not authorize the full corpus backfill. Human review of this report remains the approval boundary.',
    '',
  );
  return `${lines.join('\n')}\n`;
}

const RECALL_QUALITY_WORK_DIRECTORY_NAME_FOR_REPORT = 'recall-quality-evaluation';
