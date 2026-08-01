import { basename } from 'node:path';

import { RecallSearchScope } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type {
  RecallQualityCaseId,
  RecallQualityEvaluationCase,
  RecallQualityExpectedSource,
} from './recall-quality-corpus.js';
import type { RecallConversationSearchResult } from './recall-conversation-service.js';

/** One ordered hybrid search plus its measured total-query latency. */
export interface RecallQualitySearchObservation {
  evaluationCase: RecallQualityEvaluationCase;
  results: readonly RecallConversationSearchResult[];
  searchPolicy: {
    scope: RecallSearchScope;
    invocationProjectIdentity: string | null;
  };
  globalControlResults?: readonly RecallConversationSearchResult[];
  queryLatencyMilliseconds: number;
}

/** Median and nearest-rank p95 latency in milliseconds. */
export interface RecallQualityLatencySummary {
  median: number;
  p95: number;
}

/** Per-case outcome at one final-result count. */
export interface RecallQualityCaseFinalMeasurement {
  finalCount: number;
  finalRecalled: boolean;
  contextUseful: boolean;
  sourceOccurrencesPreserved: boolean;
  preservedSourceOccurrences: number;
  sessionOriginsVerified: boolean;
  evidenceRelationsVerified: boolean;
  contributingEntriesVerified: boolean;
  branchesVerified: boolean;
  finalDuplicateSlots: number;
  finalResultSlots: number;
}

/** One channel's proof that global pollution displaced project evidence before its limit. */
export interface RecallChannelLimitMeasurement {
  channel: 'dense' | 'lexical' | 'identifier';
  projectSourceAdmitted: boolean;
  globalSourceDisplaced: boolean;
  pollutingCandidateCount: number;
  passed: boolean;
}

/** Source, scope, exclusion, duplicate, context, and latency evidence for one fixed query. */
export interface RecallQualityCaseMeasurement {
  caseId: RecallQualityCaseId;
  category: RecallQualityEvaluationCase['category'];
  scope: RecallSearchScope;
  searchScopeVerified: boolean;
  invocationProjectIdentityVerified: boolean;
  excludedSessionFilesAbsent: boolean;
  preLimitChannelsVerified: boolean;
  preLimitChannelMeasurements: RecallChannelLimitMeasurement[];
  candidatePoolRecalled: boolean;
  rawCandidateCount: number;
  groupedCandidateCount: number;
  candidatePoolDuplicateSlots: number;
  queryLatencyMilliseconds: number;
  finalCounts: RecallQualityCaseFinalMeasurement[];
}

/** Aggregate ordered hybrid quality at one final-result count. */
export interface RecallQualityFinalCountMeasurement {
  finalCount: number;
  finalRecall: number;
  contextUsefulness: number;
  sourceOccurrencePreservation: number;
  sessionOriginVerification: number;
  evidenceRelationVerification: number;
  contributingEntryVerification: number;
  branchVerification: number;
  finalDuplicateRate: number;
  missedCaseIds: RecallQualityCaseId[];
  contextFailureCaseIds: RecallQualityCaseId[];
  sourceOccurrenceFailureCaseIds: RecallQualityCaseId[];
  sessionOriginFailureCaseIds: RecallQualityCaseId[];
  evidenceRelationFailureCaseIds: RecallQualityCaseId[];
  contributingEntryFailureCaseIds: RecallQualityCaseId[];
  branchFailureCaseIds: RecallQualityCaseId[];
  finalDuplicateSlots: number;
  finalResultSlots: number;
}

/** Complete measured quality and latency for one candidate-count search configuration. */
export interface RecallQualityMeasurement {
  caseCount: number;
  candidatePoolRecall: number;
  candidatePoolDuplicateRate: number;
  queryLatencyMilliseconds: RecallQualityLatencySummary;
  queryLatencyByScope: {
    project: RecallQualityLatencySummary | null;
    global: RecallQualityLatencySummary | null;
  };
  policyFailureCaseIds: RecallQualityCaseId[];
  missedCandidatePoolCaseIds: RecallQualityCaseId[];
  caseMeasurements: RecallQualityCaseMeasurement[];
  finalCounts: RecallQualityFinalCountMeasurement[];
}

function getRecallResultGroupMembers(result: RecallConversationSearchResult): RecallSearchResult[] {
  return [result, ...result.duplicateOccurrences];
}

function hasExpectedEntryId(
  candidate: RecallSearchResult,
  expectedSource: RecallQualityExpectedSource,
): boolean {
  return (
    candidate.entryId.value === expectedSource.entryId ||
    candidate.contributingEntryIds.some(({ value }) => value === expectedSource.entryId)
  );
}

function matchesExpectedRecallSource(
  candidate: RecallSearchResult,
  expectedSource: RecallQualityExpectedSource,
): boolean {
  if (
    basename(candidate.sessionPath) !== expectedSource.sessionFile ||
    !hasExpectedEntryId(candidate, expectedSource) ||
    !expectedSource.requiredText.every((requiredText) => candidate.content.includes(requiredText))
  ) {
    return false;
  }
  if (
    expectedSource.expectedEvidenceKind &&
    candidate.evidenceKind !== expectedSource.expectedEvidenceKind
  ) {
    return false;
  }
  return !(
    expectedSource.expectedSummaryKind &&
    candidate.summaryKind !== expectedSource.expectedSummaryKind
  );
}

function matchesExpectedRecallBranch(
  candidate: RecallSearchResult,
  expectedSource: RecallQualityExpectedSource,
): boolean {
  return !(
    (expectedSource.expectedBranch === 'active' && !candidate.isOnActiveBranch) ||
    (expectedSource.expectedBranch === 'abandoned' && candidate.isOnActiveBranch)
  );
}

function verifiesEveryExpectedSource(
  results: readonly RecallConversationSearchResult[],
  evaluationCase: RecallQualityEvaluationCase,
  verify: (
    result: RecallConversationSearchResult,
    candidate: RecallSearchResult,
    expectedSource: RecallQualityExpectedSource,
  ) => boolean,
): boolean {
  return evaluationCase.expectedSources.every((expectedSource) =>
    results.some((result) =>
      getRecallResultGroupMembers(result).some(
        (candidate) =>
          matchesExpectedRecallSource(candidate, expectedSource) &&
          verify(result, candidate, expectedSource),
      ),
    ),
  );
}

function resultGroupMatchesEvaluationCase(
  result: RecallConversationSearchResult,
  evaluationCase: RecallQualityEvaluationCase,
): boolean {
  return getRecallResultGroupMembers(result).some((candidate) =>
    evaluationCase.expectedSources.some((expectedSource) =>
      matchesExpectedRecallSource(candidate, expectedSource),
    ),
  );
}

function countPreservedExpectedSources(
  results: readonly RecallConversationSearchResult[],
  evaluationCase: RecallQualityEvaluationCase,
): number {
  const candidates = results.flatMap(getRecallResultGroupMembers);
  return evaluationCase.expectedSources.filter((expectedSource) =>
    candidates.some((candidate) => matchesExpectedRecallSource(candidate, expectedSource)),
  ).length;
}

function hasUsefulRecallContext(
  results: readonly RecallConversationSearchResult[],
  evaluationCase: RecallQualityEvaluationCase,
): boolean {
  const requiredFragments = evaluationCase.requiredContext.map((fragment) =>
    fragment.toLowerCase(),
  );
  return results.some((result) => {
    if (!resultGroupMatchesEvaluationCase(result, evaluationCase)) {
      return false;
    }
    const displayedContent = (result.neighborContext?.content ?? result.content).toLowerCase();
    return requiredFragments.every((fragment) => displayedContent.includes(fragment));
  });
}

function haveMatchingContributingEntryIds(
  left: RecallSearchResult,
  right: RecallSearchResult,
): boolean {
  return (
    left.contributingEntryIds.length === right.contributingEntryIds.length &&
    left.contributingEntryIds.every(
      ({ value }, index) => value === right.contributingEntryIds[index]?.value,
    )
  );
}

// Keep this evaluation oracle independent from production grouping so one shared bug cannot
// make duplicate suppression and the duplicate-rate metric agree incorrectly.
function areExactCrossSessionCopiesForEvaluation(
  left: RecallSearchResult,
  right: RecallSearchResult,
): boolean {
  return (
    left.sessionPath !== right.sessionPath &&
    left.checksum === right.checksum &&
    left.content === right.content &&
    left.documentKind === right.documentKind &&
    left.summaryKind === right.summaryKind &&
    left.evidenceKind === right.evidenceKind &&
    left.evidencePart === right.evidencePart &&
    left.role === right.role
  );
}

function areOverlappingSourceSpans(left: RecallSearchResult, right: RecallSearchResult): boolean {
  return (
    left.sessionPath === right.sessionPath &&
    left.entryId.value === right.entryId.value &&
    haveMatchingContributingEntryIds(left, right) &&
    left.documentKind === right.documentKind &&
    left.summaryKind === right.summaryKind &&
    left.evidenceKind === right.evidenceKind &&
    left.evidencePart === right.evidencePart &&
    left.role === right.role &&
    left.textRunId === right.textRunId &&
    left.characterStart < right.characterEnd &&
    right.characterStart < left.characterEnd
  );
}

function countDuplicateRecallSlots(candidates: readonly RecallSearchResult[]): number {
  let duplicateSlots = 0;
  for (const [index, candidate] of candidates.entries()) {
    const earlierCandidates = candidates.slice(0, index);
    if (
      earlierCandidates.some(
        (earlier) =>
          areExactCrossSessionCopiesForEvaluation(candidate, earlier) ||
          areOverlappingSourceSpans(candidate, earlier),
      )
    ) {
      duplicateSlots += 1;
    }
  }
  return duplicateSlots;
}

function createRate(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function findFailureCaseIds(
  outcomes: ReadonlyArray<{
    caseId: RecallQualityCaseId;
    outcome: RecallQualityCaseFinalMeasurement;
  }>,
  isFailure: (outcome: RecallQualityCaseFinalMeasurement) => boolean,
): RecallQualityCaseId[] {
  return outcomes.filter(({ outcome }) => isFailure(outcome)).map(({ caseId }) => caseId);
}

function readLatencyPercentile(values: readonly number[], percentile: number): number {
  const sorted = values.toSorted((left, right) => left - right);
  const index = Math.max(0, Math.ceil(sorted.length * percentile) - 1);
  const value = sorted[index];
  if (value === undefined) {
    throw new Error('Recall quality latency percentile requires at least one observation');
  }
  return value;
}

function summarizeRecallLatency(values: readonly number[]): RecallQualityLatencySummary {
  return {
    median: readLatencyPercentile(values, 0.5),
    p95: readLatencyPercentile(values, 0.95),
  };
}

function assertRecallQualityInputs(
  observations: readonly RecallQualitySearchObservation[],
  finalCounts: readonly number[],
): void {
  if (observations.length === 0) {
    throw new Error('Recall quality measurement requires at least one search observation');
  }
  if (finalCounts.length === 0 || new Set(finalCounts).size !== finalCounts.length) {
    throw new Error('Recall quality measurement requires unique final-result counts');
  }
  for (const [index, finalCount] of finalCounts.entries()) {
    if (
      !Number.isInteger(finalCount) ||
      finalCount < 1 ||
      finalCount > 200 ||
      (index > 0 && finalCount <= (finalCounts[index - 1] ?? 0))
    ) {
      throw new Error(
        'Recall quality measurement final-result counts must be ascending integers from 1 to 200',
      );
    }
  }
  const caseIds = observations.map(({ evaluationCase }) => evaluationCase.id);
  if (new Set(caseIds).size !== caseIds.length) {
    throw new Error('Recall quality measurement requires one observation per case id');
  }
  for (const observation of observations) {
    if (
      !Number.isFinite(observation.queryLatencyMilliseconds) ||
      observation.queryLatencyMilliseconds < 0
    ) {
      throw new Error('Recall quality measurement latency must be a finite nonnegative number');
    }
  }
}

function hasRecallCandidateChannel(
  candidate: RecallSearchResult,
  channel: RecallChannelLimitMeasurement['channel'],
): boolean {
  return candidate[channel] !== null;
}

function measurePreLimitChannels(
  observation: RecallQualitySearchObservation,
): RecallChannelLimitMeasurement[] {
  const proof = observation.evaluationCase.preLimitChannelProof;
  if (!proof) {
    return [];
  }
  const globalCandidates = observation.globalControlResults?.flatMap(getRecallResultGroupMembers);
  if (!globalCandidates) {
    return proof.requiredChannels.map((channel) => ({
      channel,
      projectSourceAdmitted: false,
      globalSourceDisplaced: false,
      pollutingCandidateCount: 0,
      passed: false,
    }));
  }
  const projectCandidates = observation.results.flatMap(getRecallResultGroupMembers);
  return proof.requiredChannels.map((channel) => {
    const projectSourceAdmitted = observation.evaluationCase.expectedSources.every(
      (expectedSource) =>
        projectCandidates.some(
          (candidate) =>
            matchesExpectedRecallSource(candidate, expectedSource) &&
            hasRecallCandidateChannel(candidate, channel),
        ),
    );
    const globalSourceDisplaced = observation.evaluationCase.expectedSources.every(
      (expectedSource) =>
        !globalCandidates.some(
          (candidate) =>
            matchesExpectedRecallSource(candidate, expectedSource) &&
            hasRecallCandidateChannel(candidate, channel),
        ),
    );
    const pollutingCandidateCount = globalCandidates.filter(
      (candidate) =>
        proof.pollutingSessionFiles.includes(basename(candidate.sessionPath)) &&
        hasRecallCandidateChannel(candidate, channel),
    ).length;
    return {
      channel,
      projectSourceAdmitted,
      globalSourceDisplaced,
      pollutingCandidateCount,
      passed:
        projectSourceAdmitted &&
        globalSourceDisplaced &&
        pollutingCandidateCount >= proof.minimumPollutingCandidatesPerChannel,
    };
  });
}

function measureRecallQualityCase(
  observation: RecallQualitySearchObservation,
  finalCounts: readonly number[],
): RecallQualityCaseMeasurement {
  const rawCandidates = observation.results.flatMap(getRecallResultGroupMembers);
  const preLimitChannelMeasurements = measurePreLimitChannels(observation);
  const expectedInvocationProjectIdentity =
    observation.evaluationCase.expectedInvocationProjectIdentity ?? null;
  return {
    caseId: observation.evaluationCase.id,
    category: observation.evaluationCase.category,
    scope: observation.evaluationCase.scope,
    searchScopeVerified: observation.searchPolicy.scope === observation.evaluationCase.scope,
    invocationProjectIdentityVerified:
      observation.searchPolicy.invocationProjectIdentity === expectedInvocationProjectIdentity,
    excludedSessionFilesAbsent: rawCandidates.every(
      (candidate) =>
        !observation.evaluationCase.excludedSessionFiles.includes(basename(candidate.sessionPath)),
    ),
    preLimitChannelsVerified: preLimitChannelMeasurements.every(({ passed }) => passed),
    preLimitChannelMeasurements,
    candidatePoolRecalled: rawCandidates.some((candidate) =>
      observation.evaluationCase.expectedSources.some((expectedSource) =>
        matchesExpectedRecallSource(candidate, expectedSource),
      ),
    ),
    rawCandidateCount: rawCandidates.length,
    groupedCandidateCount: observation.results.length,
    candidatePoolDuplicateSlots: countDuplicateRecallSlots(rawCandidates),
    queryLatencyMilliseconds: observation.queryLatencyMilliseconds,
    finalCounts: finalCounts.map((finalCount) => {
      const results = observation.results.slice(0, finalCount);
      const preservedSourceOccurrences = countPreservedExpectedSources(
        results,
        observation.evaluationCase,
      );
      return {
        finalCount,
        finalRecalled: results.some((result) =>
          resultGroupMatchesEvaluationCase(result, observation.evaluationCase),
        ),
        contextUseful: hasUsefulRecallContext(results, observation.evaluationCase),
        sourceOccurrencesPreserved:
          preservedSourceOccurrences >=
          observation.evaluationCase.minimumPreservedSourceOccurrences,
        preservedSourceOccurrences,
        sessionOriginsVerified: verifiesEveryExpectedSource(
          results,
          observation.evaluationCase,
          (result, candidate, expectedSource) =>
            candidate.cwd === expectedSource.expectedSessionOrigin,
        ),
        evidenceRelationsVerified: verifiesEveryExpectedSource(
          results,
          observation.evaluationCase,
          (result, candidate, expectedSource) =>
            result.evidenceRelation === expectedSource.expectedEvidenceRelation,
        ),
        contributingEntriesVerified: verifiesEveryExpectedSource(
          results,
          observation.evaluationCase,
          (result, candidate, expectedSource) =>
            expectedSource.requiredContributingEntryIds.every((requiredEntryId) =>
              candidate.contributingEntryIds.some(({ value }) => value === requiredEntryId),
            ),
        ),
        branchesVerified: verifiesEveryExpectedSource(
          results,
          observation.evaluationCase,
          (result, candidate, expectedSource) =>
            matchesExpectedRecallBranch(candidate, expectedSource),
        ),
        finalDuplicateSlots: countDuplicateRecallSlots(results),
        finalResultSlots: results.length,
      };
    }),
  };
}

/** Measures candidate-pool and ordered final recall, context, provenance, duplicates, and latency. */
export function measureRecallQuality(
  observations: readonly RecallQualitySearchObservation[],
  finalCounts: readonly number[],
): RecallQualityMeasurement {
  assertRecallQualityInputs(observations, finalCounts);
  const caseMeasurements = observations.map((observation) =>
    measureRecallQualityCase(observation, finalCounts),
  );
  const missedCandidatePoolCaseIds = caseMeasurements
    .filter(({ candidatePoolRecalled }) => !candidatePoolRecalled)
    .map(({ caseId }) => caseId);
  const rawCandidateCount = caseMeasurements.reduce(
    (total, measurement) => total + measurement.rawCandidateCount,
    0,
  );
  const candidatePoolDuplicateSlots = caseMeasurements.reduce(
    (total, measurement) => total + measurement.candidatePoolDuplicateSlots,
    0,
  );
  const policyFailureCaseIds = caseMeasurements
    .filter(
      (measurement) =>
        !measurement.searchScopeVerified ||
        !measurement.invocationProjectIdentityVerified ||
        !measurement.excludedSessionFilesAbsent ||
        !measurement.preLimitChannelsVerified,
    )
    .map(({ caseId }) => caseId);
  const latencyByScope = (scope: RecallSearchScope): RecallQualityLatencySummary | null => {
    const values = caseMeasurements
      .filter((measurement) => measurement.scope === scope)
      .map(({ queryLatencyMilliseconds }) => queryLatencyMilliseconds);
    return values.length > 0 ? summarizeRecallLatency(values) : null;
  };
  return {
    caseCount: caseMeasurements.length,
    candidatePoolRecall: createRate(
      caseMeasurements.length - missedCandidatePoolCaseIds.length,
      caseMeasurements.length,
    ),
    candidatePoolDuplicateRate: createRate(candidatePoolDuplicateSlots, rawCandidateCount),
    queryLatencyMilliseconds: summarizeRecallLatency(
      caseMeasurements.map(({ queryLatencyMilliseconds }) => queryLatencyMilliseconds),
    ),
    queryLatencyByScope: {
      project: latencyByScope(RecallSearchScope.PROJECT),
      global: latencyByScope(RecallSearchScope.GLOBAL),
    },
    policyFailureCaseIds,
    missedCandidatePoolCaseIds,
    caseMeasurements,
    finalCounts: finalCounts.map((finalCount) => {
      const outcomes = caseMeasurements.map((measurement) => {
        const outcome = measurement.finalCounts.find(
          (candidate) => candidate.finalCount === finalCount,
        );
        if (!outcome) {
          throw new Error(
            `Recall quality measurement missing final-result outcome for ${measurement.caseId}@${finalCount}`,
          );
        }
        return { caseId: measurement.caseId, outcome };
      });
      const missedCaseIds = findFailureCaseIds(outcomes, (outcome) => !outcome.finalRecalled);
      const contextFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.contextUseful,
      );
      const sourceOccurrenceFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.sourceOccurrencesPreserved,
      );
      const sessionOriginFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.sessionOriginsVerified,
      );
      const evidenceRelationFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.evidenceRelationsVerified,
      );
      const contributingEntryFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.contributingEntriesVerified,
      );
      const branchFailureCaseIds = findFailureCaseIds(
        outcomes,
        (outcome) => !outcome.branchesVerified,
      );
      const finalDuplicateSlots = outcomes.reduce(
        (total, { outcome }) => total + outcome.finalDuplicateSlots,
        0,
      );
      const finalResultSlots = outcomes.reduce(
        (total, { outcome }) => total + outcome.finalResultSlots,
        0,
      );
      return {
        finalCount,
        finalRecall: createRate(outcomes.length - missedCaseIds.length, outcomes.length),
        contextUsefulness: createRate(
          outcomes.length - contextFailureCaseIds.length,
          outcomes.length,
        ),
        sourceOccurrencePreservation: createRate(
          outcomes.length - sourceOccurrenceFailureCaseIds.length,
          outcomes.length,
        ),
        sessionOriginVerification: createRate(
          outcomes.length - sessionOriginFailureCaseIds.length,
          outcomes.length,
        ),
        evidenceRelationVerification: createRate(
          outcomes.length - evidenceRelationFailureCaseIds.length,
          outcomes.length,
        ),
        contributingEntryVerification: createRate(
          outcomes.length - contributingEntryFailureCaseIds.length,
          outcomes.length,
        ),
        branchVerification: createRate(
          outcomes.length - branchFailureCaseIds.length,
          outcomes.length,
        ),
        finalDuplicateRate: createRate(finalDuplicateSlots, finalResultSlots),
        missedCaseIds,
        contextFailureCaseIds,
        sourceOccurrenceFailureCaseIds,
        sessionOriginFailureCaseIds,
        evidenceRelationFailureCaseIds,
        contributingEntryFailureCaseIds,
        branchFailureCaseIds,
        finalDuplicateSlots,
        finalResultSlots,
      };
    }),
  };
}
