import { cpus, type CpuInfo } from 'node:os';

import { createCanonicalIdentity } from './create-canonical-identity.js';
import { EmbeddedInferenceComputeBackend, type RecallInferenceBackend } from './enums.js';
import type {
  RecallQueryPlanningModelProfile,
  RecallRerankingModelProfile,
} from './recall-model-profiles.js';

/** Normalizes resolved physical device names for stable physical device-bound execution identity. */
export function normalizeRecallPhysicalDeviceIdentity(
  deviceNames: readonly string[],
): readonly string[] {
  const normalizedNames = deviceNames
    .map((deviceName) => deviceName.trim().replace(/\s+/gu, ' ').toLocaleLowerCase('en-US'))
    .filter(Boolean);
  return Object.freeze([...new Set(normalizedNames)].sort());
}

/** Resolves publishable CPU names plus a stable model-and-logical-processor physical device identity. */
export function resolveRecallCpuPhysicalDeviceIdentity(
  processors: readonly Pick<CpuInfo, 'model'>[] = cpus(),
): { deviceNames: readonly string[]; physicalDeviceIdentity: readonly string[] } {
  const deviceNamesByNormalizedName = new Map<string, string>();
  for (const processor of processors) {
    const displayName = processor.model.trim().replace(/\s+/gu, ' ');
    if (displayName) {
      deviceNamesByNormalizedName.set(displayName.toLocaleLowerCase('en-US'), displayName);
    }
  }
  const deviceNames = [...deviceNamesByNormalizedName.values()].sort((left, right) =>
    left.localeCompare(right, 'en-US'),
  );
  if (deviceNames.length === 0) {
    deviceNames.push('Unknown CPU');
  }
  return Object.freeze({
    deviceNames: Object.freeze(deviceNames),
    physicalDeviceIdentity: normalizeRecallPhysicalDeviceIdentity([
      ...deviceNames,
      `logical-processors:${processors.length}`,
    ]),
  });
}

/** Resolves published names and physical device identity for one selected compute backend. */
export function resolveRecallPhysicalDeviceIdentity(
  computeBackend: EmbeddedInferenceComputeBackend,
  acceleratedDeviceNames?: readonly string[],
): { deviceNames: readonly string[]; physicalDeviceIdentity: readonly string[] } {
  if (computeBackend === EmbeddedInferenceComputeBackend.CPU) {
    return resolveRecallCpuPhysicalDeviceIdentity();
  }
  const deviceNames = Object.freeze([...(acceleratedDeviceNames ?? [computeBackend])]);
  return Object.freeze({
    deviceNames,
    physicalDeviceIdentity: normalizeRecallPhysicalDeviceIdentity(deviceNames),
  });
}

/** Embeds recall queries and index documents through explicitly distinct model operations. */
export interface RecallEmbeddingProvider {
  /** Embeds one submitted recall query using the model profile's query semantics. */
  embedQuery(query: string, signal?: AbortSignal): Promise<number[]>;

  /** Embeds index documents in input order using the model profile's document semantics. */
  embedDocuments(documents: readonly string[], signal?: AbortSignal): Promise<number[][]>;
}

/** Produces one finite relevance score per candidate document in input order. */
export interface RecallRerankingProvider {
  /** Scores candidate documents against one query without reordering the candidates. */
  rerankDocuments(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<number[]>;
}

/** One typed lexical, semantic, or hypothetical-answer query in an ordered query plan. */
export interface RecallPlannedRetrievalQuery {
  type: 'lex' | 'vec' | 'hyde';
  query: string;
}

/** Submitted recall query plus optional intent passed only to the query planning model. */
export interface RecallQueryPlanningRequest {
  query: string;
  recallIntent?: string;
}

/** Produces one bounded ordered query plan without executing retrieval or ranking. */
export interface RecallQueryPlanningProvider {
  /** Plans lexical, semantic, and optional hypothetical-answer retrieval queries. */
  planRecallQuery(
    request: Readonly<RecallQueryPlanningRequest>,
    signal?: AbortSignal,
  ): Promise<RecallPlannedRetrievalQuery[]>;
}

/** Inspectable adapter, policy, timeout, and cache identity for query planning execution. */
export interface RecallQueryPlanningExecutionIdentity {
  adapterId: string;
  adapterVersion: string;
  /** Opaque identity for endpoint or device settings that can change generated plans. */
  adapterConfigurationIdentity: string;
  backend: RecallInferenceBackend;
  cacheIdentity: string;
  modelProfileId: string;
  modelProfileIdentity: string;
  promptPolicy: string;
  grammarVersion: string;
  requestTimeoutMilliseconds: number;
}

/** Query planning provider whose profile and adapter identity can be verified before use. */
export interface RecallIdentifiedQueryPlanningProvider extends RecallQueryPlanningProvider {
  readonly executionIdentity: Readonly<RecallQueryPlanningExecutionIdentity>;
}

/** Creates query planner identity from model, adapter, endpoint or device, prompt, and grammar policy. */
export function createRecallQueryPlanningExecutionIdentity(
  profile: RecallQueryPlanningModelProfile,
  adapterId: string,
  adapterConfigurationIdentity: string,
  backend: RecallQueryPlanningExecutionIdentity['backend'],
  requestTimeoutMilliseconds: number,
  adapterVersion: string = adapterId,
): Readonly<RecallQueryPlanningExecutionIdentity> {
  const modelProfileIdentity = createCanonicalIdentity(
    'recall-query-planning-model-profile-v1',
    profile,
  );
  return Object.freeze({
    adapterId,
    adapterVersion,
    adapterConfigurationIdentity,
    backend,
    cacheIdentity: createCanonicalIdentity('recall-query-planning-execution-v1', {
      adapterConfigurationIdentity,
      adapterId,
      adapterVersion,
      backend,
      modelProfileIdentity,
      requestTimeoutMilliseconds,
    }),
    modelProfileId: profile.profileId,
    modelProfileIdentity,
    promptPolicy: profile.promptPolicy,
    grammarVersion: profile.grammarVersion,
    requestTimeoutMilliseconds,
  });
}

/** Search and cache identity for one reranking adapter executing one model profile. */
export interface RecallRerankingExecutionIdentity {
  adapterId: string;
  adapterVersion: string;
  adapterConfigurationIdentity: string;
  backend: RecallInferenceBackend;
  cacheIdentity: string;
  modelProfileId: string;
  modelProfileIdentity: string;
  requestTimeoutMilliseconds: number;
}

/** Reranking provider whose profile and adapter identity can be verified before use. */
export interface RecallIdentifiedRerankingProvider extends RecallRerankingProvider {
  readonly executionIdentity: Readonly<RecallRerankingExecutionIdentity>;
}

/** Creates cache identity that changes with either reranker profile or adapter policy. */
export function createRecallRerankingExecutionIdentity(
  profile: RecallRerankingModelProfile,
  adapterId: string,
  adapterConfigurationIdentity: string,
  backend: RecallRerankingExecutionIdentity['backend'],
  requestTimeoutMilliseconds: number,
  adapterVersion: string = adapterId,
): Readonly<RecallRerankingExecutionIdentity> {
  const modelProfileIdentity = createCanonicalIdentity(
    'recall-reranking-model-profile-v1',
    profile,
  );
  return Object.freeze({
    adapterId,
    adapterVersion,
    adapterConfigurationIdentity,
    backend,
    cacheIdentity: createCanonicalIdentity('recall-reranking-execution-v1', {
      adapterConfigurationIdentity,
      adapterId,
      adapterVersion,
      backend,
      modelProfileIdentity,
      requestTimeoutMilliseconds,
    }),
    modelProfileId: profile.profileId,
    modelProfileIdentity,
    requestTimeoutMilliseconds,
  });
}
