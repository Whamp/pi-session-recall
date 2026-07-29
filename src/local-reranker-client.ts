import { Type } from 'typebox';
import { Value } from 'typebox/value';

/** Bounds a local reranker request to 60 seconds when the caller supplies no timeout. */
export const DEFAULT_LOCAL_RERANKER_REQUEST_TIMEOUT_MILLISECONDS = 60_000;

const LOCAL_RERANKER_RESPONSE_SCHEMA = Type.Object({
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

/** OpenAI-compatible endpoint and model settings for the local Qwen reranker. */
export interface LocalRerankerClientConfig {
  baseUrl: string;
  model: string;
  requestTimeoutMilliseconds?: number;
}

/** Scores candidate documents against one query in the candidates' original order. */
export interface LocalRerankerClient {
  rerankDocuments(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<number[]>;
}

/** Normalizes one HTTP reranker base URL to the exact request endpoint used by the adapter. */
export function normalizeLocalRerankerEndpoint(baseUrl: string): string {
  if (!URL.canParse(baseUrl)) {
    throw new Error(`Recall reranker base URL invalid: ${baseUrl}`);
  }
  const parsed = new URL(baseUrl);
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Recall reranker base URL invalid protocol: ${parsed.protocol}`);
  }
  parsed.pathname = `${parsed.pathname.replace(/\/+$/u, '')}/rerank`;
  return parsed.toString();
}

/** Creates a validated OpenAI-compatible client for the locally served Qwen reranker. */
export function createLocalRerankerClient(config: LocalRerankerClientConfig): LocalRerankerClient {
  const endpoint = normalizeLocalRerankerEndpoint(config.baseUrl);
  const model = config.model.trim();
  if (!model) {
    throw new Error('Recall reranker model invalid: expected a non-blank model name');
  }
  const requestTimeoutMilliseconds =
    config.requestTimeoutMilliseconds ?? DEFAULT_LOCAL_RERANKER_REQUEST_TIMEOUT_MILLISECONDS;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall reranker request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }

  return {
    async rerankDocuments(query, documents, signal) {
      const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
      const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            model,
            query,
            documents,
            'top_n': documents.length,
          }),
          signal: requestSignal,
        });
      } catch (error) {
        if (timeoutSignal.aborted && !signal?.aborted) {
          throw new Error(
            `Recall reranker request timed out after ${requestTimeoutMilliseconds} ms at ${endpoint}`,
            { cause: error },
          );
        }
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall reranker request failed at ${endpoint}: ${message}`, {
          cause: error,
        });
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(
          `Recall reranker request failed (${response.status}): ${body.slice(0, 500)}`,
        );
      }
      let payload: ReturnType<typeof Value.Parse<typeof LOCAL_RERANKER_RESPONSE_SCHEMA>>;
      try {
        const rawPayload: unknown = await response.json();
        payload = Value.Parse(LOCAL_RERANKER_RESPONSE_SCHEMA, rawPayload);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall reranker response invalid: ${message}`, { cause: error });
      }
      if (payload.results.length !== documents.length) {
        throw new Error(
          `Recall reranker response count mismatch: expected ${documents.length}, received ${payload.results.length}`,
        );
      }
      const scoresByIndex = new Map<number, number>();
      for (const result of payload.results) {
        if (result.index >= documents.length) {
          throw new Error(
            `Recall reranker response candidate index out of range: ${result.index} for ${documents.length} documents`,
          );
        }
        if (scoresByIndex.has(result.index)) {
          throw new Error(`Recall reranker response duplicate candidate index ${result.index}`);
        }
        scoresByIndex.set(result.index, result['relevance_score']);
      }
      const orderedScores: number[] = [];
      for (let index = 0; index < documents.length; index += 1) {
        const score = scoresByIndex.get(index);
        if (score === undefined) {
          throw new Error(`Recall reranker response missing candidate index ${index}`);
        }
        orderedScores.push(score);
      }
      return orderedScores;
    },
  };
}
