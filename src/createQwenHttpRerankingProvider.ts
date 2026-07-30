import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createCanonicalIdentity } from './create-canonical-identity.js';
import { RecallInferenceBackend } from './enums.js';
import {
  createRecallRerankingExecutionIdentity,
  type RecallIdentifiedRerankingProvider,
} from './recall-inference-capabilities.js';
import type { QwenRerankingModelProfile } from './recall-model-profiles.js';

const DEFAULT_QWEN_HTTP_RERANKING_REQUEST_TIMEOUT_MILLISECONDS = 60_000;
const LLAMA_CPP_HTTP_RERANKING_ADAPTER_ID = 'llama-cpp-http-reranking-v1';
const LLAMA_CPP_HTTP_RERANKING_ADAPTER_VERSION = '1';

const qwenHttpRerankingResponseSchema = Type.Object({
  model: Type.String({ minLength: 1 }),
  object: Type.Literal('list'),
  usage: Type.Object({
    'prompt_tokens': Type.Integer({ minimum: 0 }),
    'total_tokens': Type.Integer({ minimum: 0 }),
  }),
  results: Type.Array(
    Type.Object({
      index: Type.Integer({ minimum: 0 }),
      'relevance_score': Type.Number(),
    }),
  ),
});

/** HTTP execution settings that do not contribute to the Qwen model profile identity. */
export interface QwenHttpRerankingBackendConfig {
  baseUrl: string;
  requestTimeoutMilliseconds?: number;
}

/** Creates the Qwen HTTP adapter that returns finite relevance scores in candidate order. */
export function createQwenHttpRerankingProvider(
  profile: QwenRerankingModelProfile,
  backend: QwenHttpRerankingBackendConfig,
): RecallIdentifiedRerankingProvider {
  if (!URL.canParse(backend.baseUrl)) {
    throw new Error(`Recall Qwen HTTP reranking base URL invalid: ${backend.baseUrl}`);
  }
  const parsedEndpoint = new URL(backend.baseUrl);
  if (parsedEndpoint.protocol !== 'http:' && parsedEndpoint.protocol !== 'https:') {
    throw new Error(
      `Recall Qwen HTTP reranking base URL invalid protocol: ${parsedEndpoint.protocol}`,
    );
  }
  parsedEndpoint.pathname = `${parsedEndpoint.pathname.replace(/\/+$/u, '')}/rerank`;
  const endpoint = parsedEndpoint.toString();
  const requestTimeoutMilliseconds =
    backend.requestTimeoutMilliseconds ?? DEFAULT_QWEN_HTTP_RERANKING_REQUEST_TIMEOUT_MILLISECONDS;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall Qwen HTTP reranking request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }

  return {
    executionIdentity: createRecallRerankingExecutionIdentity(
      profile,
      LLAMA_CPP_HTTP_RERANKING_ADAPTER_ID,
      createCanonicalIdentity('llama-cpp-http-reranking-config-v1', {
        normalizedEndpoint: endpoint,
        requestTimeoutMilliseconds,
      }),
      RecallInferenceBackend.LLAMA_CPP_HTTP,
      requestTimeoutMilliseconds,
      LLAMA_CPP_HTTP_RERANKING_ADAPTER_VERSION,
    ),
    async rerankDocuments(query, documents, signal) {
      const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
      const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            model: profile.model,
            query,
            documents,
            'top_n': documents.length,
          }),
          signal: requestSignal,
        });
      } catch (error) {
        if (timeoutSignal.aborted && !signal?.aborted) {
          throw new Error(
            `Recall Qwen HTTP reranking request timed out after ${requestTimeoutMilliseconds} ms at ${endpoint}`,
            { cause: error },
          );
        }
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall Qwen HTTP reranking request failed at ${endpoint}: ${message}`, {
          cause: error,
        });
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(
          `Recall Qwen HTTP reranking request failed (${response.status}): ${body.slice(0, 500)}`,
        );
      }
      let payload: ReturnType<typeof Value.Parse<typeof qwenHttpRerankingResponseSchema>>;
      try {
        const rawPayload: unknown = await response.json();
        payload = Value.Parse(qwenHttpRerankingResponseSchema, rawPayload);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall Qwen HTTP reranking response invalid: ${message}`, {
          cause: error,
        });
      }
      if (payload.results.length !== documents.length) {
        throw new Error(
          `Recall Qwen HTTP reranking response count mismatch: expected ${documents.length}, received ${payload.results.length}`,
        );
      }
      const scoresByIndex = new Map<number, number>();
      for (const result of payload.results) {
        if (result.index >= documents.length) {
          throw new Error(
            `Recall Qwen HTTP reranking response candidate index out of range: ${result.index} for ${documents.length} documents`,
          );
        }
        if (scoresByIndex.has(result.index)) {
          throw new Error(
            `Recall Qwen HTTP reranking response duplicate candidate index ${result.index}`,
          );
        }
        const score = result['relevance_score'];
        if (!Number.isFinite(score)) {
          throw new Error(
            `Recall Qwen HTTP reranking response score invalid at candidate index ${result.index}`,
          );
        }
        if (score < profile.scoreRange.minimum || score > profile.scoreRange.maximum) {
          throw new Error(
            `Recall Qwen HTTP reranker score outside profile range at candidate index ${result.index}: expected ${profile.scoreRange.minimum} through ${profile.scoreRange.maximum}, received ${score}`,
          );
        }
        scoresByIndex.set(result.index, score);
      }
      return Array.from(documents.keys(), (index) => {
        const score = scoresByIndex.get(index);
        if (score === undefined) {
          throw new Error(`Recall Qwen HTTP reranking response missing candidate index ${index}`);
        }
        return score;
      });
    },
  };
}
