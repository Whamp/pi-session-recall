import { Type } from 'typebox';
import { Value } from 'typebox/value';

const DEFAULT_LOCAL_EMBEDDING_REQUEST_TIMEOUT_MILLISECONDS = 60_000;

const localEmbeddingResponseSchema = Type.Object({
  data: Type.Array(
    Type.Object({
      index: Type.Optional(Type.Integer({ minimum: 0 })),
      embedding: Type.Array(Type.Number()),
    }),
  ),
});

/** OpenAI-compatible endpoint settings, including the exact vector width required by zvec. */
export interface LocalEmbeddingClientConfig {
  baseUrl: string;
  model: string;
  dimensions: number;
  batchSize?: number;
  requestTimeoutMilliseconds?: number;
}

/** Batch embedding capability used for both stored conversation text and recall queries. */
export interface LocalEmbeddingClient {
  embedTexts(texts: string[], signal?: AbortSignal): Promise<number[][]>;
}

/** Creates an OpenAI-compatible client for a locally served embedding model. */
export function createLocalEmbeddingClient(
  config: LocalEmbeddingClientConfig,
): LocalEmbeddingClient {
  const batchSize = config.batchSize ?? 16;
  const requestTimeoutMilliseconds =
    config.requestTimeoutMilliseconds ?? DEFAULT_LOCAL_EMBEDDING_REQUEST_TIMEOUT_MILLISECONDS;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall embedding request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }
  const endpoint = `${config.baseUrl.replace(/\/$/, '')}/embeddings`;

  return {
    async embedTexts(texts, signal) {
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
            body: JSON.stringify({ model: config.model, input }),
            signal: requestSignal,
          });
        } catch (error) {
          if (timeoutSignal.aborted && !signal?.aborted) {
            throw new Error(
              `Recall embedding request timed out after ${requestTimeoutMilliseconds} ms at ${endpoint}`,
              { cause: error },
            );
          }
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Recall embedding request failed at ${endpoint}: ${message}`, {
            cause: error,
          });
        }
        if (!response.ok) {
          const body = await response.text();
          throw new Error(
            `Recall embedding request failed (${response.status}): ${body.slice(0, 500)}`,
          );
        }
        let payload: ReturnType<typeof Value.Parse<typeof localEmbeddingResponseSchema>>;
        try {
          const rawPayload: unknown = await response.json();
          payload = Value.Parse(localEmbeddingResponseSchema, rawPayload);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Recall embedding response invalid: ${message}`, { cause: error });
        }
        const ordered = [...payload.data].sort(
          (left, right) => (left.index ?? 0) - (right.index ?? 0),
        );
        if (ordered.length !== input.length) {
          throw new Error(
            `Recall embedding response count mismatch: expected ${input.length}, received ${ordered.length}`,
          );
        }
        for (const item of ordered) {
          const embedding = item.embedding;
          if (!embedding || embedding.length !== config.dimensions) {
            throw new Error(
              `Recall embedding dimension mismatch: expected ${config.dimensions}, received ${embedding?.length ?? 0}`,
            );
          }
          vectors.push(embedding);
        }
      }
      return vectors;
    },
  };
}
