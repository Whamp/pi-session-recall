import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecommendedOptionalInferenceCandidates,
  readRecommendedOptionalInferenceConformance,
} from './createRecommendedOptionalInferenceCandidates.js';

void test('production setup reads independently accepted reranking evidence', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-optional-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidencePath = join(root, 'inference-conformance.json');
  await writeFile(
    evidencePath,
    `${JSON.stringify({
      reranking: {
        query: 'source provenance',
        documents: ['Preserve source provenance.', 'Change the navigation color.'],
        expectedScores: [0.9, 0.1],
        maximumAbsoluteDifference: 1e-6,
      },
      queryPlanning: null,
    })}\n`,
    'utf8',
  );

  assert.deepEqual(await readRecommendedOptionalInferenceConformance(evidencePath), {
    rerankingConformance: {
      query: 'source provenance',
      documents: ['Preserve source provenance.', 'Change the navigation color.'],
      expectedScores: [0.9, 0.1],
      maximumAbsoluteDifference: 1e-6,
    },
  });
});

void test('production HTTP reranking and query-planning candidates run live conformance', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-optional-candidates-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const requestedPaths: string[] = [];
  const server = createServer((request, response) => {
    requestedPaths.push(request.url ?? 'missing');
    response.setHeader('content-type', 'application/json');
    if (request.url === '/v1/rerank') {
      response.end(
        JSON.stringify({
          model: 'qwen3-rerank',
          object: 'list',
          usage: { prompt_tokens: 4, total_tokens: 4 },
          results: [
            { index: 0, relevance_score: 0.9 },
            { index: 1, relevance_score: 0.1 },
          ],
        }),
      );
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.end(
        JSON.stringify({
          model: 'qmd-query-expansion-1.7B-q4_k_m',
          choices: [
            {
              message: {
                role: 'assistant',
                content: [
                  'lex: source provenance evidence',
                  'vec: source provenance in recalled conversations',
                ].join('\n'),
              },
            },
          ],
        }),
      );
      return;
    }
    response.statusCode = 404;
    response.end('{}');
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(
    () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        });
      }),
  );
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const baseUrl = `http://127.0.0.1:${address.port}/v1`;
  const config = await loadRecallConversationConfig({
    homeDirectory: root,
    environment: {
      PI_RECALL_RERANKER_BASE_URL: baseUrl,
      PI_RECALL_QUERY_PLANNER_BASE_URL: baseUrl,
    },
  });
  const candidates = createRecommendedOptionalInferenceCandidates(config, {
    rerankingConformance: {
      query: 'source provenance',
      documents: ['Preserve source provenance.', 'Change the navigation color.'],
      expectedScores: [0.9, 0.1],
    },
  });
  const reranking = candidates.find(
    ({ candidateId }) => candidateId === 'recommended-qwen-reranker-http',
  );
  const queryPlanning = candidates.find(
    ({ candidateId }) => candidateId === 'recommended-qmd-query-planner-http',
  );
  assert.ok(reranking);
  assert.ok(queryPlanning);

  const rerankingResult = await reranking.verifyCapabilityConformance();
  const queryPlanningResult = await queryPlanning.verifyCapabilityConformance();

  assert.equal(rerankingResult.profileId, 'qwen3-reranker-0.6b-q8-0-v1');
  assert.equal(queryPlanningResult.profileId, 'qmd-query-expansion-1.7b-q4-k-m-v1');
  assert.deepEqual(requestedPaths, ['/v1/rerank', '/v1/chat/completions']);
});
