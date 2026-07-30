import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service joins target dense and lexical candidates by evidence occurrence', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-hybrid-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'hybrid.jsonl');
  await writeFile(
    sessionPath,
    `${JSON.stringify({ type: 'session', version: 3, id: 'hybrid', timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'entry', parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: 'hybrid ownership sentinel' } })}\n`,
  );
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_hybrid_owner';
  await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sessionPath],
  });
  const results = await service.searchRecallGenerationHybrid(
    generationId,
    'hybrid ownership sentinel',
    5,
  );
  assert.equal(results[0]?.evidence.content, 'hybrid ownership sentinel');
  assert.equal(results[0]?.lexicalRank, 1);
  assert.ok(results[0]?.denseRank !== null);
});
