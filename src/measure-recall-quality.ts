import { basename } from 'node:path';

import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type {
  RecallQualityEvaluationCase,
  RecallQualityExpectedSource,
} from './recall-quality-corpus.js';
import type { RerankedRecallSearchResult } from './rerank-recall-search-results.js';

/** One fully reranked search plus separately measured total-query and reranker latency. */
export interface RecallQualitySearchObservation {
  evaluationCase: RecallQualityEvaluationCase;
  results: readonly RerankedRecallSearchResult[];
  queryLatencyMilliseconds: number;
  rerankerLatencyMilliseconds: number;
}

/** Median and nearest-rank p95 latency in milliseconds. */
export interface RecallQualityLatencySummary {
  median: number;
  p95: number;
}

/** Per-case outcome at one final-result count. */
export interface RecallQualityCaseFinalMeasurement {
  finalCount: number;
  postRerankRecalled: boolean;
  contextUseful: boolean;
  sourceOccurrencesPreserved: boolean;
  preservedSourceOccurrences: number;
  postRerankDuplicateSlots: number;
  postRerankResultSlots: number;
}

/** Source, duplicate, context, and latency evidence for one fixed query. */
export interface RecallQualityCaseMeasurement {
  caseId: string;
  category: RecallQualityEvaluationCase['category'];
  preRerankRecalled: boolean;
  rawCandidateCount: number;
  groupedCandidateCount: number;
  preRerankDuplicateSlots: number;
  queryLatencyMilliseconds: number;
  rerankerLatencyMilliseconds: number;
  finalCounts: RecallQualityCaseFinalMeasurement[];
}

/** Aggregate post-rerank quality at one final-result count. */
export interface RecallQualityFinalCountMeasurement {
  finalCount: number;
  postRerankRecall: number;
  contextUsefulness: number;
  sourceOccurrencePreservation: number;
  postRerankDuplicateRate: number;
  missedCaseIds: string[];
  contextFailureCaseIds: string[];
  sourceOccurrenceFailureCaseIds: string[];
  postRerankDuplicateSlots: number;
  postRerankResultSlots: number;
}

/** Complete measured quality and latency for one candidate-count search configuration. */
export interface RecallQualityMeasurement {
  caseCount: number;
  preRerankRecall: number;
  preRerankDuplicateRate: number;
  queryLatencyMilliseconds: RecallQualityLatencySummary;
  rerankerLatencyMilliseconds: RecallQualityLatencySummary;
  missedPreRerankCaseIds: string[];
  caseMeasurements: RecallQualityCaseMeasurement[];
  finalCounts: RecallQualityFinalCountMeasurement[];
}

function getRecallResultGroupMembers(result: RerankedRecallSearchResult): RecallSearchResult[] {
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
    !hasExpectedEntryId(candidate, expectedSource)
  ) {
    return false;
  }
  if (
    expectedSource.expectedEvidenceKind &&
    candidate.evidenceKind !== expectedSource.expectedEvidenceKind
  ) {
    return false;
  }
  if (
    expectedSource.expectedSummaryKind &&
    candidate.summaryKind !== expectedSource.expectedSummaryKind
  ) {
    return false;
  }
  if (expectedSource.expectedBranch === 'active' && !candidate.isOnActiveBranch) {
    return false;
  }
  if (expectedSource.expectedBranch === 'abandoned' && candidate.isOnActiveBranch) {
    return false;
  }
  return true;
}

function resultGroupMatchesEvaluationCase(
  result: RerankedRecallSearchResult,
  evaluationCase: RecallQualityEvaluationCase,
): boolean {
  return getRecallResultGroupMembers(result).some((candidate) =>
    evaluationCase.expectedSources.some((expectedSource) =>
      matchesExpectedRecallSource(candidate, expectedSource),
    ),
  );
}

function countPreservedExpectedSources(
  results: readonly RerankedRecallSearchResult[],
  evaluationCase: RecallQualityEvaluationCase,
): number {
  const candidates = results.flatMap(getRecallResultGroupMembers);
  return evaluationCase.expectedSources.filter((expectedSource) =>
    candidates.some((candidate) => matchesExpectedRecallSource(candidate, expectedSource)),
  ).length;
}

function hasUsefulRecallContext(
  results: readonly RerankedRecallSearchResult[],
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

function areExactCrossSessionCopies(left: RecallSearchResult, right: RecallSearchResult): boolean {
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
          areExactCrossSessionCopies(candidate, earlier) ||
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
      observation.queryLatencyMilliseconds < 0 ||
      !Number.isFinite(observation.rerankerLatencyMilliseconds) ||
      observation.rerankerLatencyMilliseconds < 0
    ) {
      throw new Error('Recall quality measurement latency must be a finite nonnegative number');
    }
  }
}

function measureRecallQualityCase(
  observation: RecallQualitySearchObservation,
  finalCounts: readonly number[],
): RecallQualityCaseMeasurement {
  const rawCandidates = observation.results.flatMap(getRecallResultGroupMembers);
  return {
    caseId: observation.evaluationCase.id,
    category: observation.evaluationCase.category,
    preRerankRecalled: rawCandidates.some((candidate) =>
      observation.evaluationCase.expectedSources.some((expectedSource) =>
        matchesExpectedRecallSource(candidate, expectedSource),
      ),
    ),
    rawCandidateCount: rawCandidates.length,
    groupedCandidateCount: observation.results.length,
    preRerankDuplicateSlots: countDuplicateRecallSlots(rawCandidates),
    queryLatencyMilliseconds: observation.queryLatencyMilliseconds,
    rerankerLatencyMilliseconds: observation.rerankerLatencyMilliseconds,
    finalCounts: finalCounts.map((finalCount) => {
      const results = observation.results.slice(0, finalCount);
      const preservedSourceOccurrences = countPreservedExpectedSources(
        results,
        observation.evaluationCase,
      );
      return {
        finalCount,
        postRerankRecalled: results.some((result) =>
          resultGroupMatchesEvaluationCase(result, observation.evaluationCase),
        ),
        contextUseful: hasUsefulRecallContext(results, observation.evaluationCase),
        sourceOccurrencesPreserved:
          preservedSourceOccurrences >=
          observation.evaluationCase.minimumPreservedSourceOccurrences,
        preservedSourceOccurrences,
        postRerankDuplicateSlots: countDuplicateRecallSlots(results),
        postRerankResultSlots: results.length,
      };
    }),
  };
}

/** Measures pre/post-rerank recall, duplicates, context, source preservation, and latency. */
export function measureRecallQuality(
  observations: readonly RecallQualitySearchObservation[],
  finalCounts: readonly number[],
): RecallQualityMeasurement {
  assertRecallQualityInputs(observations, finalCounts);
  const caseMeasurements = observations.map((observation) =>
    measureRecallQualityCase(observation, finalCounts),
  );
  const missedPreRerankCaseIds = caseMeasurements
    .filter(({ preRerankRecalled }) => !preRerankRecalled)
    .map(({ caseId }) => caseId);
  const rawCandidateCount = caseMeasurements.reduce(
    (total, measurement) => total + measurement.rawCandidateCount,
    0,
  );
  const preRerankDuplicateSlots = caseMeasurements.reduce(
    (total, measurement) => total + measurement.preRerankDuplicateSlots,
    0,
  );
  return {
    caseCount: caseMeasurements.length,
    preRerankRecall: createRate(
      caseMeasurements.length - missedPreRerankCaseIds.length,
      caseMeasurements.length,
    ),
    preRerankDuplicateRate: createRate(preRerankDuplicateSlots, rawCandidateCount),
    queryLatencyMilliseconds: summarizeRecallLatency(
      caseMeasurements.map(({ queryLatencyMilliseconds }) => queryLatencyMilliseconds),
    ),
    rerankerLatencyMilliseconds: summarizeRecallLatency(
      caseMeasurements.map(({ rerankerLatencyMilliseconds }) => rerankerLatencyMilliseconds),
    ),
    missedPreRerankCaseIds,
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
      const missedCaseIds = outcomes
        .filter(({ outcome }) => !outcome.postRerankRecalled)
        .map(({ caseId }) => caseId);
      const contextFailureCaseIds = outcomes
        .filter(({ outcome }) => !outcome.contextUseful)
        .map(({ caseId }) => caseId);
      const sourceOccurrenceFailureCaseIds = outcomes
        .filter(({ outcome }) => !outcome.sourceOccurrencesPreserved)
        .map(({ caseId }) => caseId);
      const postRerankDuplicateSlots = outcomes.reduce(
        (total, { outcome }) => total + outcome.postRerankDuplicateSlots,
        0,
      );
      const postRerankResultSlots = outcomes.reduce(
        (total, { outcome }) => total + outcome.postRerankResultSlots,
        0,
      );
      return {
        finalCount,
        postRerankRecall: createRate(outcomes.length - missedCaseIds.length, outcomes.length),
        contextUsefulness: createRate(
          outcomes.length - contextFailureCaseIds.length,
          outcomes.length,
        ),
        sourceOccurrencePreservation: createRate(
          outcomes.length - sourceOccurrenceFailureCaseIds.length,
          outcomes.length,
        ),
        postRerankDuplicateRate: createRate(postRerankDuplicateSlots, postRerankResultSlots),
        missedCaseIds,
        contextFailureCaseIds,
        sourceOccurrenceFailureCaseIds,
        postRerankDuplicateSlots,
        postRerankResultSlots,
      };
    }),
  };
}
