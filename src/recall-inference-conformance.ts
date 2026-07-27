import type {
  RecallEmbeddingProvider,
  RecallRerankingProvider,
} from './recall-inference-capabilities.js';
import type {
  RecallEmbeddingModelProfile,
  RecallRerankingModelProfile,
} from './recall-model-profiles.js';

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
  provider: RecallRerankingProvider;
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
