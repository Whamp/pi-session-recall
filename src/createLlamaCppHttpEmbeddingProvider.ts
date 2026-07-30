import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';

const DEFAULT_LLAMA_CPP_HTTP_EMBEDDING_REQUEST_TIMEOUT_MILLISECONDS = 60_000;

const llamaCppHttpEmbeddingResponseSchema = Type.Object({
  data: Type.Array(
    Type.Object({
      index: Type.Optional(Type.Integer({ minimum: 0 })),
      embedding: Type.Array(Type.Number()),
    }),
  ),
});

/** llama.cpp HTTP execution settings excluded from compatible embedding model semantics. */
export interface LlamaCppHttpEmbeddingBackendConfig {
  baseUrl: string;
  batchSize?: number;
  requestTimeoutMilliseconds?: number;
}

/** Creates a capability-specific llama.cpp HTTP adapter with profile-owned input prompts. */
export function createLlamaCppHttpEmbeddingProvider(
  profile: RecallEmbeddingModelProfile,
  backend: LlamaCppHttpEmbeddingBackendConfig,
): RecallEmbeddingProvider {
  const batchSize = backend.batchSize ?? 16;
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error(
      `Recall llama.cpp HTTP embedding batch size invalid: expected a positive integer, received ${batchSize}`,
    );
  }
  const requestTimeoutMilliseconds =
    backend.requestTimeoutMilliseconds ??
    DEFAULT_LLAMA_CPP_HTTP_EMBEDDING_REQUEST_TIMEOUT_MILLISECONDS;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall llama.cpp HTTP embedding request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }
  if (!URL.canParse(backend.baseUrl)) {
    throw new Error(`Recall llama.cpp HTTP embedding base URL invalid: ${backend.baseUrl}`);
  }
  const parsedEndpoint = new URL(backend.baseUrl);
  if (parsedEndpoint.protocol !== 'http:' && parsedEndpoint.protocol !== 'https:') {
    throw new Error(
      `Recall llama.cpp HTTP embedding base URL invalid protocol: ${parsedEndpoint.protocol}`,
    );
  }
  parsedEndpoint.pathname = `${parsedEndpoint.pathname.replace(/\/+$/u, '')}/embeddings`;
  const endpoint = parsedEndpoint.toString();

  async function embedLlamaCppHttpTexts(
    texts: readonly string[],
    signal?: AbortSignal,
  ): Promise<number[][]> {
    const vectors: number[][] = [];
    for (let start = 0; start < texts.length; start += batchSize) {
      const input = texts.slice(start, start + batchSize);
      const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
      const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model: profile.identity.requestModel, input }),
          signal: requestSignal,
        });
      } catch (error) {
        if (timeoutSignal.aborted && !signal?.aborted) {
          throw new Error(
            `Recall llama.cpp HTTP embedding request timed out after ${requestTimeoutMilliseconds} ms at ${endpoint}`,
            { cause: error },
          );
        }
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Recall llama.cpp HTTP embedding request failed at ${endpoint}: ${message}`,
          {
            cause: error,
          },
        );
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(
          `Recall llama.cpp HTTP embedding request failed (${response.status}): ${body.slice(0, 500)}`,
        );
      }
      let payload: ReturnType<typeof Value.Parse<typeof llamaCppHttpEmbeddingResponseSchema>>;
      try {
        const rawPayload: unknown = await response.json();
        payload = Value.Parse(llamaCppHttpEmbeddingResponseSchema, rawPayload);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall llama.cpp HTTP embedding response invalid: ${message}`, {
          cause: error,
        });
      }
      const ordered = [...payload.data].sort(
        (left, right) => (left.index ?? 0) - (right.index ?? 0),
      );
      if (ordered.length !== input.length) {
        throw new Error(
          `Recall llama.cpp HTTP embedding response count mismatch: expected ${input.length}, received ${ordered.length}`,
        );
      }
      for (const item of ordered) {
        if (item.embedding.length !== profile.identity.dimensions) {
          throw new Error(
            `Recall llama.cpp HTTP embedding dimension mismatch: expected ${profile.identity.dimensions}, received ${item.embedding.length}`,
          );
        }
        vectors.push(item.embedding);
      }
    }
    return vectors;
  }

  return {
    async embedQuery(query, signal) {
      const embeddings = await embedLlamaCppHttpTexts(
        [`${profile.queryInputPrefix}${query}`],
        signal,
      );
      const embedding = embeddings[0];
      if (!embedding) {
        throw new Error('Recall llama.cpp HTTP embedding response missing query vector');
      }
      return embedding;
    },
    embedDocuments(documents, signal) {
      return embedLlamaCppHttpTexts(
        documents.map((document) => `${profile.documentInputPrefix}${document}`),
        signal,
      );
    },
  };
}
