import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  createRecallQueryPlanningExecutionIdentity,
  type RecallIdentifiedQueryPlanningProvider,
} from './recall-inference-capabilities.js';
import { RecallInferenceBackend } from './enums.js';
import type { RecallQueryPlanningModelProfile } from './recall-model-profiles.js';
import {
  formatQmdQueryPlanningPrompt,
  parseQmdQueryPlanningOutput,
} from './recall-query-planning-policy.js';

const DEFAULT_QMD_HTTP_QUERY_PLANNING_TIMEOUT_MILLISECONDS = 60_000;

const QMD_HTTP_QUERY_PLANNING_RESPONSE_SCHEMA = Type.Object({
  model: Type.String({ minLength: 1 }),
  choices: Type.Array(
    Type.Object({
      message: Type.Object({
        role: Type.Literal('assistant'),
        content: Type.String(),
        reasoning_content: Type.Optional(Type.String()),
      }),
    }),
    { minItems: 1, maxItems: 1 },
  ),
  usage: Type.Optional(
    Type.Object({
      prompt_tokens: Type.Integer({ minimum: 0 }),
      completion_tokens: Type.Integer({ minimum: 0 }),
      total_tokens: Type.Integer({ minimum: 0 }),
    }),
  ),
});

/** HTTP endpoint and timeout settings outside QMD query planner profile semantics. */
export interface QmdHttpQueryPlanningBackendConfig {
  baseUrl: string;
  requestTimeoutMilliseconds?: number;
}

function createQmdHttpQueryPlanningEndpoint(baseUrl: string): string {
  if (!URL.canParse(baseUrl)) {
    throw new Error(`Recall QMD query planner base URL invalid: ${baseUrl}`);
  }
  const parsed = new URL(baseUrl);
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Recall QMD query planner base URL invalid protocol: ${parsed.protocol}`);
  }
  parsed.pathname = `${parsed.pathname.replace(/\/+$/u, '')}/chat/completions`;
  return parsed.toString();
}

/** Creates the capability-specific llama.cpp HTTP adapter for QMD query planning. */
export function createQmdHttpQueryPlanningProvider(
  profile: RecallQueryPlanningModelProfile,
  backend: QmdHttpQueryPlanningBackendConfig,
): RecallIdentifiedQueryPlanningProvider {
  const endpoint = createQmdHttpQueryPlanningEndpoint(backend.baseUrl);
  const requestTimeoutMilliseconds =
    backend.requestTimeoutMilliseconds ?? DEFAULT_QMD_HTTP_QUERY_PLANNING_TIMEOUT_MILLISECONDS;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall QMD query planner request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }
  return {
    executionIdentity: createRecallQueryPlanningExecutionIdentity(
      profile,
      'llama-cpp-http-query-planning-v1',
      RecallInferenceBackend.LLAMA_CPP_HTTP,
      requestTimeoutMilliseconds,
    ),
    async planRecallQuery(request, signal) {
      const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
      const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            model: profile.model,
            messages: [
              {
                role: 'user',
                content: formatQmdQueryPlanningPrompt(request.query, request.recallIntent),
              },
            ],
            grammar: profile.grammar,
            max_tokens: profile.generationPolicy.maximumOutputTokens,
            temperature: profile.generationPolicy.temperature,
            top_k: profile.generationPolicy.topK,
            top_p: profile.generationPolicy.topP,
            repeat_last_n: profile.generationPolicy.repeatPenaltyLastTokens,
            presence_penalty: profile.generationPolicy.presencePenalty,
            stream: false,
          }),
          signal: requestSignal,
        });
      } catch (error) {
        if (timeoutSignal.aborted && !signal?.aborted) {
          throw new Error(
            `Recall QMD query planner request timed out after ${requestTimeoutMilliseconds} ms at ${endpoint}`,
            { cause: error },
          );
        }
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall QMD query planner request failed at ${endpoint}: ${message}`, {
          cause: error,
        });
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(
          `Recall QMD query planner request failed (${response.status}): ${body.slice(0, 500)}`,
        );
      }
      let payload: ReturnType<typeof Value.Parse<typeof QMD_HTTP_QUERY_PLANNING_RESPONSE_SCHEMA>>;
      try {
        const rawPayload: unknown = await response.json();
        payload = Value.Parse(QMD_HTTP_QUERY_PLANNING_RESPONSE_SCHEMA, rawPayload);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall QMD query planner response invalid: ${message}`, { cause: error });
      }
      if (payload.model !== profile.model) {
        throw new Error(
          `Recall QMD query planner response model mismatch: expected ${profile.model}, received ${payload.model}`,
        );
      }
      const message = payload.choices[0]?.message;
      const generatedPlan = message?.content.trim() || message?.reasoning_content?.trim();
      if (!generatedPlan) {
        throw new Error(
          'Recall QMD query planner response invalid: missing assistant content and reasoning content',
        );
      }
      return parseQmdQueryPlanningOutput(generatedPlan, profile);
    },
  };
}
