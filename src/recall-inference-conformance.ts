import type {
  RecallEmbeddingProvider,
  RecallIdentifiedQueryPlanningProvider,
  RecallIdentifiedRerankingProvider,
  RecallPlannedRetrievalQuery,
} from './recall-inference-capabilities.js';
import type {
  RecallEmbeddingModelProfile,
  RecallQueryPlanningModelProfile,
  RecallRerankingModelProfile,
} from './recall-model-profiles.js';
import { validateQmdQueryPlanningPlan } from './recall-query-planning-policy.js';

/** Deterministic probe and expected vectors for one embedding provider conformance run. */
export interface RecallEmbeddingProviderConformanceOptions {
  provider: RecallEmbeddingProvider;
  profile: RecallEmbeddingModelProfile;
  query: string;
  expectedQueryEmbedding: readonly number[];
  documents: readonly string[];
  expectedDocumentEmbeddings: readonly (readonly number[])[];
  maximumAbsoluteDifference?: number;
  monotonicMilliseconds?: () => number;
  signal?: AbortSignal;
}

/** Bounded capability counts and timings measured by embedding provider conformance. */
export interface RecallEmbeddingProviderConformanceMeasurement {
  queryCount: 1;
  documentCount: number;
  queryMilliseconds: number;
  documentMilliseconds: number;
}

/** Deterministic probe and expected ordered scores for one reranking conformance run. */
export interface RecallRerankingProviderConformanceOptions {
  provider: RecallIdentifiedRerankingProvider;
  profile: RecallRerankingModelProfile;
  query: string;
  documents: readonly string[];
  expectedScores: readonly number[];
  maximumAbsoluteDifference?: number;
  monotonicMilliseconds?: () => number;
  signal?: AbortSignal;
}

/** Bounded capability counts and timing measured by reranking provider conformance. */
export interface RecallRerankingProviderConformanceMeasurement {
  queryCount: 1;
  documentCount: number;
  rerankingMilliseconds: number;
}

/** Deterministic prompt, protected terms, and expected plan for query planner conformance. */
export interface RecallQueryPlanningProviderConformanceOptions {
  provider: RecallIdentifiedQueryPlanningProvider;
  profile: RecallQueryPlanningModelProfile;
  query: string;
  recallIntent?: string;
  protectedTerms: readonly string[];
  expectedPlan?: readonly Readonly<RecallPlannedRetrievalQuery>[];
  monotonicMilliseconds?: () => number;
  signal?: AbortSignal;
}

/** Bounded typed-query counts and timing measured by query planner conformance. */
export interface RecallQueryPlanningProviderConformanceMeasurement {
  plannedQueryCount: number;
  lexQueryCount: number;
  vecQueryCount: number;
  hydeQueryCount: number;
  planningMilliseconds: number;
}

function assertConformanceVector(
  capability: string,
  actual: readonly number[],
  expected: readonly number[],
  dimensions: number,
  maximumAbsoluteDifference: number,
  normalization?: 'l2',
): void {
  if (actual.length !== dimensions) {
    throw new Error(
      `Recall ${capability} conformance dimension mismatch: expected ${dimensions}, received ${actual.length}`,
    );
  }
  if (expected.length !== dimensions) {
    throw new Error(
      `Recall ${capability} conformance fixture dimension mismatch: expected ${dimensions}, received ${expected.length}`,
    );
  }
  let actualSquaredNorm = 0;
  for (let index = 0; index < dimensions; index += 1) {
    const actualValue = actual[index];
    const expectedValue = expected[index];
    if (actualValue === undefined || !Number.isFinite(actualValue)) {
      throw new Error(
        `Recall ${capability} conformance vector invalid at dimension ${index}: expected a finite number`,
      );
    }
    actualSquaredNorm += actualValue * actualValue;
    if (expectedValue === undefined || !Number.isFinite(expectedValue)) {
      throw new Error(
        `Recall ${capability} conformance fixture invalid at dimension ${index}: expected a finite number`,
      );
    }
    if (Math.abs(actualValue - expectedValue) > maximumAbsoluteDifference) {
      throw new Error(
        `Recall ${capability} conformance vector mismatch at dimension ${index}: expected ${expectedValue}, received ${actualValue}`,
      );
    }
  }
  if (normalization === 'l2') {
    const norm = Math.sqrt(actualSquaredNorm);
    if (Math.abs(norm - 1) > 1e-5) {
      throw new Error(
        `Recall ${capability} conformance normalization mismatch: expected L2 norm 1, received ${norm}`,
      );
    }
  }
}

/** Verifies and times distinct query/document embedding semantics against fixed vectors. */
export async function measureRecallEmbeddingProviderConformance(
  options: RecallEmbeddingProviderConformanceOptions,
): Promise<RecallEmbeddingProviderConformanceMeasurement> {
  if (options.documents.length !== options.expectedDocumentEmbeddings.length) {
    throw new Error(
      `Recall document embedding conformance fixture count mismatch: expected ${options.documents.length}, received ${options.expectedDocumentEmbeddings.length}`,
    );
  }
  const maximumAbsoluteDifference = options.maximumAbsoluteDifference ?? 0;
  if (!Number.isFinite(maximumAbsoluteDifference) || maximumAbsoluteDifference < 0) {
    throw new Error(
      'Recall embedding conformance tolerance invalid: expected a finite nonnegative number',
    );
  }
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const queryStartedAtMilliseconds = monotonicMilliseconds();
  const queryEmbedding = await options.provider.embedQuery(options.query, options.signal);
  const queryMilliseconds = Math.max(monotonicMilliseconds() - queryStartedAtMilliseconds, 0);
  assertConformanceVector(
    'query embedding',
    queryEmbedding,
    options.expectedQueryEmbedding,
    options.profile.identity.dimensions,
    maximumAbsoluteDifference,
    options.profile.identity.normalization,
  );

  const documentsStartedAtMilliseconds = monotonicMilliseconds();
  const documentEmbeddings = await options.provider.embedDocuments(
    options.documents,
    options.signal,
  );
  const documentMilliseconds = Math.max(
    monotonicMilliseconds() - documentsStartedAtMilliseconds,
    0,
  );
  if (documentEmbeddings.length !== options.documents.length) {
    throw new Error(
      `Recall document embedding conformance response count mismatch: expected ${options.documents.length}, received ${documentEmbeddings.length}`,
    );
  }
  for (let index = 0; index < documentEmbeddings.length; index += 1) {
    const actual = documentEmbeddings[index];
    const expected = options.expectedDocumentEmbeddings[index];
    if (!actual || !expected) {
      throw new Error(`Recall document embedding conformance missing vector at index ${index}`);
    }
    assertConformanceVector(
      `document embedding index ${index}`,
      actual,
      expected,
      options.profile.identity.dimensions,
      maximumAbsoluteDifference,
      options.profile.identity.normalization,
    );
  }
  return {
    queryCount: 1,
    documentCount: options.documents.length,
    queryMilliseconds,
    documentMilliseconds,
  };
}

/** Verifies and times ordered finite reranker scores against a fixed relevance fixture. */
export async function measureRecallRerankingProviderConformance(
  options: RecallRerankingProviderConformanceOptions,
): Promise<RecallRerankingProviderConformanceMeasurement> {
  if (options.profile.scoreMeaning !== 'higher-is-more-relevant') {
    throw new Error(
      `Recall reranking conformance score meaning unsupported: ${String(options.profile.scoreMeaning)}`,
    );
  }
  if (options.provider.executionIdentity.modelProfileId !== options.profile.profileId) {
    throw new Error(
      `Recall reranking conformance profile identity mismatch: expected ${options.profile.profileId}, received ${options.provider.executionIdentity.modelProfileId}`,
    );
  }
  const expectedCacheIdentity = `${options.profile.profileId}:${options.provider.executionIdentity.adapterId}`;
  if (options.provider.executionIdentity.cacheIdentity !== expectedCacheIdentity) {
    throw new Error(
      `Recall reranking conformance cache identity mismatch: expected ${expectedCacheIdentity}, received ${options.provider.executionIdentity.cacheIdentity}`,
    );
  }
  if (options.documents.length !== options.expectedScores.length) {
    throw new Error(
      `Recall reranking conformance fixture count mismatch: expected ${options.documents.length}, received ${options.expectedScores.length}`,
    );
  }
  const maximumAbsoluteDifference = options.maximumAbsoluteDifference ?? 0;
  if (!Number.isFinite(maximumAbsoluteDifference) || maximumAbsoluteDifference < 0) {
    throw new Error(
      'Recall reranking conformance tolerance invalid: expected a finite nonnegative number',
    );
  }
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const rerankingStartedAtMilliseconds = monotonicMilliseconds();
  const scores = await options.provider.rerankDocuments(
    options.query,
    options.documents,
    options.signal,
  );
  const rerankingMilliseconds = Math.max(
    monotonicMilliseconds() - rerankingStartedAtMilliseconds,
    0,
  );
  if (scores.length !== options.documents.length) {
    throw new Error(
      `Recall reranking conformance response count mismatch: expected ${options.documents.length}, received ${scores.length}`,
    );
  }
  for (let index = 0; index < scores.length; index += 1) {
    const actualScore = scores[index];
    const expectedScore = options.expectedScores[index];
    if (actualScore === undefined || !Number.isFinite(actualScore)) {
      throw new Error(
        `Recall reranking conformance score invalid at candidate index ${index}: expected a finite number`,
      );
    }
    if (
      actualScore < options.profile.scoreRange.minimum ||
      actualScore > options.profile.scoreRange.maximum
    ) {
      throw new Error(
        `Recall reranking conformance score outside profile range at candidate index ${index}: expected ${options.profile.scoreRange.minimum} through ${options.profile.scoreRange.maximum}, received ${actualScore}`,
      );
    }
    if (expectedScore === undefined || !Number.isFinite(expectedScore)) {
      throw new Error(
        `Recall reranking conformance fixture invalid at candidate index ${index}: expected a finite number`,
      );
    }
    if (Math.abs(actualScore - expectedScore) > maximumAbsoluteDifference) {
      throw new Error(
        `Recall reranking conformance score mismatch at candidate index ${index}: expected ${expectedScore}, received ${actualScore}`,
      );
    }
  }
  return {
    queryCount: 1,
    documentCount: options.documents.length,
    rerankingMilliseconds,
  };
}

/** Verifies planner identity, intent-capable invocation, typed bounds, terms, and fixed output. */
export async function measureRecallQueryPlanningProviderConformance(
  options: RecallQueryPlanningProviderConformanceOptions,
): Promise<RecallQueryPlanningProviderConformanceMeasurement> {
  const identity = options.provider.executionIdentity;
  if (identity.modelProfileId !== options.profile.profileId) {
    throw new Error(
      `Recall query planning conformance profile identity mismatch: expected ${options.profile.profileId}, received ${identity.modelProfileId}`,
    );
  }
  if (identity.promptPolicy !== options.profile.promptPolicy) {
    throw new Error(
      `Recall query planning conformance prompt policy mismatch: expected ${options.profile.promptPolicy}, received ${identity.promptPolicy}`,
    );
  }
  if (identity.grammarVersion !== options.profile.grammarVersion) {
    throw new Error(
      `Recall query planning conformance grammar version mismatch: expected ${options.profile.grammarVersion}, received ${identity.grammarVersion}`,
    );
  }
  const expectedCacheIdentity = `${options.profile.profileId}:${identity.adapterId}:${options.profile.promptPolicy}:${options.profile.grammarVersion}:${identity.adapterConfigurationIdentity}`;
  if (identity.cacheIdentity !== expectedCacheIdentity) {
    throw new Error(
      `Recall query planning conformance cache identity mismatch: expected ${expectedCacheIdentity}, received ${identity.cacheIdentity}`,
    );
  }
  if (
    !Number.isInteger(identity.requestTimeoutMilliseconds) ||
    identity.requestTimeoutMilliseconds < 1
  ) {
    throw new Error(
      `Recall query planning conformance timeout invalid: expected a positive integer, received ${identity.requestTimeoutMilliseconds}`,
    );
  }
  const protectedTerms = options.protectedTerms.map((term) => term.trim().toLocaleLowerCase());
  if (protectedTerms.length === 0 || protectedTerms.some((term) => !term)) {
    throw new Error(
      'Recall query planning conformance protected terms invalid: expected at least one non-blank term',
    );
  }
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const planningStartedAtMilliseconds = monotonicMilliseconds();
  const plan = validateQmdQueryPlanningPlan(
    await options.provider.planRecallQuery(
      {
        query: options.query,
        ...(options.recallIntent === undefined ? {} : { recallIntent: options.recallIntent }),
      },
      options.signal,
    ),
    options.profile,
  );
  const planningMilliseconds = Math.max(monotonicMilliseconds() - planningStartedAtMilliseconds, 0);
  const lexQueryCount = plan.filter(({ type }) => type === 'lex').length;
  const vecQueryCount = plan.filter(({ type }) => type === 'vec').length;
  const hydeQueryCount = plan.filter(({ type }) => type === 'hyde').length;
  const normalizedPlanText = plan
    .map(({ query }) => query)
    .join('\n')
    .toLocaleLowerCase();
  const missingProtectedTerms = protectedTerms.filter((term) => !normalizedPlanText.includes(term));
  if (missingProtectedTerms.length > 0) {
    throw new Error(
      `Recall query planning conformance protected terms missing from plan: ${missingProtectedTerms.join(', ')}`,
    );
  }
  if (options.expectedPlan) {
    if (plan.length !== options.expectedPlan.length) {
      throw new Error(
        `Recall query planning conformance fixture count mismatch: expected ${options.expectedPlan.length}, received ${plan.length}`,
      );
    }
    for (const [index, expected] of options.expectedPlan.entries()) {
      const actual = plan[index];
      if (!actual || actual.type !== expected.type || actual.query !== expected.query) {
        throw new Error(
          `Recall query planning conformance fixture mismatch at index ${index}: expected ${expected.type}: ${expected.query}, received ${actual ? `${actual.type}: ${actual.query}` : 'missing query'}`,
        );
      }
    }
  }
  return {
    plannedQueryCount: plan.length,
    lexQueryCount,
    vecQueryCount,
    hydeQueryCount,
    planningMilliseconds,
  };
}
