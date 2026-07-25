import type {
  RecallQualityFinalCountMeasurement,
  RecallQualityMeasurement,
} from './measure-recall-quality.js';
import type { RecallQualityChunkPolicy, RecallQualityGate } from './recall-quality-corpus.js';

/** Index and search evidence for one chunk-policy and per-channel candidate count. */
export interface RecallQualityConfigurationMeasurement {
  chunkPolicy: RecallQualityChunkPolicy;
  candidateCount: number;
  totalChunks: number;
  indexLatencyMilliseconds: number;
  measurement: RecallQualityMeasurement;
}

/** One measured chunk/candidate/final combination with every frozen gate decision. */
export interface RecallQualityGateCombination {
  chunkPolicy: RecallQualityChunkPolicy;
  candidateCount: number;
  finalCount: number;
  totalChunks: number;
  indexLatencyMilliseconds: number;
  candidatePoolRecall: number;
  candidatePoolDuplicateRate: number;
  finalRecall: number;
  contextUsefulness: number;
  sourceOccurrencePreservation: number;
  finalDuplicateRate: number;
  queryLatencyMilliseconds: { median: number; p95: number };
  gatePassed: boolean;
  failures: string[];
}

/** Passing evidence-selected policy, or exact blockers from the closest measured combination. */
export interface RecallQualityPolicySelection {
  passed: boolean;
  selected: RecallQualityGateCombination | null;
  blockers: string[];
  combinations: RecallQualityGateCombination[];
}

function formatQualityRate(value: number): string {
  return value.toFixed(3);
}

function formatLatency(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function findRecallQualityGateFailures(
  configuration: RecallQualityConfigurationMeasurement,
  finalCount: RecallQualityFinalCountMeasurement,
  gate: RecallQualityGate,
): string[] {
  const failures: string[] = [];
  const { measurement } = configuration;
  if (measurement.candidatePoolRecall < gate.minimumCandidatePoolRecall) {
    failures.push(
      `candidate-pool recall ${formatQualityRate(measurement.candidatePoolRecall)} is below ${formatQualityRate(gate.minimumCandidatePoolRecall)}`,
    );
  }
  if (finalCount.finalRecall < gate.minimumFinalRecall) {
    failures.push(
      `fused top-N recall ${formatQualityRate(finalCount.finalRecall)} is below ${formatQualityRate(gate.minimumFinalRecall)}`,
    );
  }
  if (finalCount.contextUsefulness < gate.minimumContextUsefulness) {
    failures.push(
      `context usefulness ${formatQualityRate(finalCount.contextUsefulness)} is below ${formatQualityRate(gate.minimumContextUsefulness)}`,
    );
  }
  if (finalCount.sourceOccurrencePreservation < gate.minimumSourceOccurrencePreservation) {
    failures.push(
      `source occurrence preservation ${formatQualityRate(finalCount.sourceOccurrencePreservation)} is below ${formatQualityRate(gate.minimumSourceOccurrencePreservation)}`,
    );
  }
  if (finalCount.finalDuplicateRate > gate.maximumFinalDuplicateRate) {
    failures.push(
      `final duplicate rate ${formatQualityRate(finalCount.finalDuplicateRate)} exceeds ${formatQualityRate(gate.maximumFinalDuplicateRate)}`,
    );
  }
  if (measurement.queryLatencyMilliseconds.p95 > gate.maximumQueryP95Milliseconds) {
    failures.push(
      `query p95 ${formatLatency(measurement.queryLatencyMilliseconds.p95)} exceeds ${formatLatency(gate.maximumQueryP95Milliseconds)}`,
    );
  }
  return failures;
}

function createRecallQualityGateCombination(
  configuration: RecallQualityConfigurationMeasurement,
  finalCount: RecallQualityFinalCountMeasurement,
  gate: RecallQualityGate,
): RecallQualityGateCombination {
  const failures = findRecallQualityGateFailures(configuration, finalCount, gate);
  return {
    chunkPolicy: { ...configuration.chunkPolicy },
    candidateCount: configuration.candidateCount,
    finalCount: finalCount.finalCount,
    totalChunks: configuration.totalChunks,
    indexLatencyMilliseconds: configuration.indexLatencyMilliseconds,
    candidatePoolRecall: configuration.measurement.candidatePoolRecall,
    candidatePoolDuplicateRate: configuration.measurement.candidatePoolDuplicateRate,
    finalRecall: finalCount.finalRecall,
    contextUsefulness: finalCount.contextUsefulness,
    sourceOccurrencePreservation: finalCount.sourceOccurrencePreservation,
    finalDuplicateRate: finalCount.finalDuplicateRate,
    queryLatencyMilliseconds: { ...configuration.measurement.queryLatencyMilliseconds },
    gatePassed: failures.length === 0,
    failures,
  };
}

function comparePassingRecallQualityCombinations(
  left: RecallQualityGateCombination,
  right: RecallQualityGateCombination,
): number {
  return (
    left.candidateCount - right.candidateCount ||
    left.finalCount - right.finalCount ||
    left.queryLatencyMilliseconds.p95 - right.queryLatencyMilliseconds.p95 ||
    left.totalChunks - right.totalChunks ||
    left.indexLatencyMilliseconds - right.indexLatencyMilliseconds ||
    left.chunkPolicy.maxTokens - right.chunkPolicy.maxTokens
  );
}

function compareBlockedRecallQualityCombinations(
  left: RecallQualityGateCombination,
  right: RecallQualityGateCombination,
): number {
  return (
    left.failures.length - right.failures.length ||
    right.candidatePoolRecall - left.candidatePoolRecall ||
    right.finalRecall - left.finalRecall ||
    right.contextUsefulness - left.contextUsefulness ||
    right.sourceOccurrencePreservation - left.sourceOccurrencePreservation ||
    left.finalDuplicateRate - right.finalDuplicateRate ||
    comparePassingRecallQualityCombinations(left, right)
  );
}

/** Selects counts and chunk policy only from measured combinations that clear every gate. */
export function selectRecallQualityPolicy(
  configurations: readonly RecallQualityConfigurationMeasurement[],
  gate: RecallQualityGate,
): RecallQualityPolicySelection {
  const combinations = configurations.flatMap((configuration) =>
    configuration.measurement.finalCounts.map((finalCount) =>
      createRecallQualityGateCombination(configuration, finalCount, gate),
    ),
  );
  const selected = combinations
    .filter(({ gatePassed }) => gatePassed)
    .toSorted(comparePassingRecallQualityCombinations)[0];
  if (selected) {
    return { passed: true, selected, blockers: [], combinations };
  }
  const closest = combinations.toSorted(compareBlockedRecallQualityCombinations)[0];
  if (!closest) {
    return {
      passed: false,
      selected: null,
      blockers: ['No recall quality configurations were measured'],
      combinations,
    };
  }
  const configurationLabel = `${closest.chunkPolicy.id}, ${closest.candidateCount} candidates/channel, ${closest.finalCount} final`;
  return {
    passed: false,
    selected: null,
    blockers: closest.failures.map((failure) => `${configurationLabel}: ${failure}`),
    combinations,
  };
}
