import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { RecallInferenceBackend } from './enums.js';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createQmdHttpQueryPlanningProvider } from './createQmdHttpQueryPlanningProvider.js';
import { measureRecallQueryPlanningProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

const QUERY_PLANNING_REQUEST_SCHEMA = Type.Object({
  model: Type.String(),
  messages: Type.Array(
    Type.Object({
      role: Type.Literal('user'),
      content: Type.String(),
    }),
  ),
  grammar: Type.String(),
  max_tokens: Type.Integer(),
  temperature: Type.Number(),
  top_k: Type.Integer(),
  top_p: Type.Number(),
  repeat_last_n: Type.Integer(),
  presence_penalty: Type.Number(),
  stream: Type.Literal(false),
});

void test('QMD HTTP query planner passes shared bounded-plan conformance with recall intent', async (t) => {
  const requests: Array<ReturnType<typeof Value.Parse<typeof QUERY_PLANNING_REQUEST_SCHEMA>>> = [];
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const generatedOutput = [
    'hyde: Copper Finch records connect recalled recovery evidence to its Pi session location.',
    'lex: recovery session evidence',
    'lex: retained Finch recovery records',
    'vec: how Copper Finch is retained for recalled conversation evidence',
    'vec: where Finch recovery evidence connects to its original Pi session',
  ].join('\n');
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      requests.push(Value.Parse(QUERY_PLANNING_REQUEST_SCHEMA, JSON.parse(body)));
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: profile.model,
          choices: [
            {
              message: {
                role: 'assistant',
                content: '',
                reasoning_content: generatedOutput,
              },
            },
          ],
          usage: { prompt_tokens: 28, completion_tokens: 52, total_tokens: 80 },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const provider = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    requestTimeoutMilliseconds: 12_345,
  });
  const clockValues = [0, 17];

  const measurement = await measureRecallQueryPlanningProviderConformance({
    provider,
    profile,
    query: profile.conformanceCanary.query,
    recallIntent: profile.conformanceCanary.recallIntent,
    protectedTerms: profile.conformanceCanary.protectedTerms,
    expectedPlan: [
      {
        type: 'hyde',
        query:
          'Copper Finch records connect recalled recovery evidence to its Pi session location.',
      },
      { type: 'lex', query: 'recovery session evidence' },
      { type: 'lex', query: 'retained Finch recovery records' },
      {
        type: 'vec',
        query: 'how Copper Finch is retained for recalled conversation evidence',
      },
      {
        type: 'vec',
        query: 'where Finch recovery evidence connects to its original Pi session',
      },
    ],
    monotonicMilliseconds() {
      const value = clockValues.shift();
      assert.notEqual(value, undefined);
      return value ?? 0;
    },
  });

  assert.deepEqual(provider.executionIdentity, {
    adapterId: 'llama-cpp-http-query-planning-v1',
    backend: 'llama-cpp-http',
    cacheIdentity:
      'qmd-query-expansion-1.7b-q4-k-m-v1:llama-cpp-http-query-planning-v1:qmd-query-expansion-no-think-v1:qmd-bounded-query-plan-v2',
    modelProfileId: profile.profileId,
    promptPolicy: profile.promptPolicy,
    grammarVersion: profile.grammarVersion,
    requestTimeoutMilliseconds: 12_345,
  });
  assert.deepEqual(requests, [
    {
      model: profile.model,
      messages: [
        {
          role: 'user',
          content:
            '/no_think Expand this search query: Copper Finch\nQuery intent: Find Pi conversation evidence about the exact Copper Finch recovery entity.',
        },
      ],
      grammar: profile.grammar,
      max_tokens: 600,
      temperature: 0.7,
      top_k: 20,
      top_p: 0.8,
      repeat_last_n: 64,
      presence_penalty: 0.5,
      stream: false,
    },
  ]);
  assert.deepEqual(measurement, {
    plannedQueryCount: 5,
    lexQueryCount: 2,
    vecQueryCount: 2,
    hydeQueryCount: 1,
    planningMilliseconds: 17,
  });
});

void test('QMD HTTP query planner fails closed on model, grammar, timeout, and cancellation', async (t) => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  let responseMode: 'wrong-model' | 'invalid-grammar' | 'invalid-plan' | 'pending' = 'wrong-model';
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      if (responseMode === 'pending') {
        return;
      }
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: responseMode === 'wrong-model' ? 'different-query-planner' : profile.model,
          choices: [
            {
              message: {
                role: 'assistant',
                content:
                  responseMode === 'invalid-grammar'
                    ? 'source provenance without a typed prefix'
                    : responseMode === 'invalid-plan'
                      ? 'lex: source provenance only'
                      : 'lex: source provenance\nvec: source provenance evidence',
              },
            },
          ],
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const baseUrl = `http://127.0.0.1:${address.port}/v1`;
  const provider = createQmdHttpQueryPlanningProvider(profile, { baseUrl });
  const conformanceOptions = {
    provider,
    profile,
    query: profile.conformanceCanary.query,
    recallIntent: profile.conformanceCanary.recallIntent,
    protectedTerms: profile.conformanceCanary.protectedTerms,
  };

  await assert.rejects(
    () => measureRecallQueryPlanningProviderConformance(conformanceOptions),
    /Recall QMD query planner response model mismatch/u,
  );

  responseMode = 'invalid-grammar';
  await assert.rejects(
    () => measureRecallQueryPlanningProviderConformance(conformanceOptions),
    /Recall query planning output grammar invalid at line 1/u,
  );

  responseMode = 'invalid-plan';
  await assert.rejects(
    () => measureRecallQueryPlanningProviderConformance(conformanceOptions),
    /Recall query planning output bounds invalid/u,
  );

  responseMode = 'pending';
  const timeoutProvider = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl,
    requestTimeoutMilliseconds: 5,
  });
  await assert.rejects(
    () =>
      measureRecallQueryPlanningProviderConformance({
        ...conformanceOptions,
        provider: timeoutProvider,
      }),
    /Recall QMD query planner request timed out after 5 ms/u,
  );

  const cancellation = new AbortController();
  const cancellationReason = new Error('operator cancelled planner verification');
  const cancelled = measureRecallQueryPlanningProviderConformance({
    ...conformanceOptions,
    signal: cancellation.signal,
  });
  cancellation.abort(cancellationReason);
  await assert.rejects(
    () => cancelled,
    /Recall QMD query planner request failed .*operator cancelled planner verification/u,
  );
});

void test('query planning conformance rejects missing protected terms from a custom adapter', async () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const adapterId = 'unprotected-query-planning-v1';
  await assert.rejects(
    () =>
      measureRecallQueryPlanningProviderConformance({
        provider: {
          executionIdentity: {
            adapterId,
            backend: RecallInferenceBackend.CUSTOM,
            cacheIdentity: `${profile.profileId}:${adapterId}:${profile.promptPolicy}:${profile.grammarVersion}`,
            modelProfileId: profile.profileId,
            promptPolicy: profile.promptPolicy,
            grammarVersion: profile.grammarVersion,
            requestTimeoutMilliseconds: 1_000,
          },
          async planRecallQuery() {
            return [
              { type: 'lex', query: 'unrelated keywords' },
              { type: 'vec', query: 'a semantic reformulation without the protected entity' },
            ];
          },
        },
        profile,
        query: profile.conformanceCanary.query,
        recallIntent: profile.conformanceCanary.recallIntent,
        protectedTerms: profile.conformanceCanary.protectedTerms,
      }),
    /Recall query planning conformance protected terms missing from plan/u,
  );
});
