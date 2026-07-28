import type { RecallInferenceBackend } from './enums.js';
import type { RecallQueryPlanningModelProfile } from './recall-model-profiles.js';

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
  backend: RecallInferenceBackend;
  cacheIdentity: string;
  modelProfileId: string;
  promptPolicy: string;
  grammarVersion: string;
  requestTimeoutMilliseconds: number;
}

/** Query planning provider whose profile and adapter identity can be verified before use. */
export interface RecallIdentifiedQueryPlanningProvider extends RecallQueryPlanningProvider {
  readonly executionIdentity: Readonly<RecallQueryPlanningExecutionIdentity>;
}

/** Creates query planner cache identity from profile, adapter, prompt, and grammar policy. */
export function createRecallQueryPlanningExecutionIdentity(
  profile: RecallQueryPlanningModelProfile,
  adapterId: string,
  backend: RecallQueryPlanningExecutionIdentity['backend'],
  requestTimeoutMilliseconds: number,
): Readonly<RecallQueryPlanningExecutionIdentity> {
  return Object.freeze({
    adapterId,
    backend,
    cacheIdentity: `${profile.profileId}:${adapterId}:${profile.promptPolicy}:${profile.grammarVersion}`,
    modelProfileId: profile.profileId,
    promptPolicy: profile.promptPolicy,
    grammarVersion: profile.grammarVersion,
    requestTimeoutMilliseconds,
  });
}

/** Search and cache identity for one reranking adapter executing one model profile. */
export interface RecallRerankingExecutionIdentity {
  adapterId: string;
  backend: RecallInferenceBackend;
  cacheIdentity: string;
  modelProfileId: string;
}

/** Reranking provider whose profile and adapter identity can be verified before use. */
export interface RecallIdentifiedRerankingProvider extends RecallRerankingProvider {
  readonly executionIdentity: Readonly<RecallRerankingExecutionIdentity>;
}

/** Creates cache identity that changes with either reranker profile or adapter policy. */
export function createRecallRerankingExecutionIdentity(
  modelProfileId: string,
  adapterId: string,
  backend: RecallRerankingExecutionIdentity['backend'],
): Readonly<RecallRerankingExecutionIdentity> {
  return Object.freeze({
    adapterId,
    backend,
    cacheIdentity: `${modelProfileId}:${adapterId}`,
    modelProfileId,
  });
}
