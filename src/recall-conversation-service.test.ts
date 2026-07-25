import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { createRecallConversationService } from './recall-conversation-service.js';

void test('recall service incrementally indexes sessions and returns source-backed semantic matches', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-1',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'queue-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'We chose a durable queue for job delivery.' }],
        },
      },
      {
        type: 'message',
        id: 'ui-entry',
        parentId: 'queue-entry',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'The navigation bar is blue.' }],
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const embeddedInputs: string[] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddedInputs.push(...texts);
      return texts.map((text) => (text.toLowerCase().includes('queue') ? [1, 0, 0] : [0, 1, 0]));
    },
  };
  const service = createRecallConversationService(
    {
      sessionsDirectory,
      databasePath: join(directory, 'zvec'),
      statePath: join(directory, 'state.json'),
      lockPath: join(directory, 'recall.lock'),
      embeddingBaseUrl: 'http://unused.test/v1',
      embeddingModel: 'test',
      embeddingDimensions: 3,
      embeddingBatchSize: 8,
    },
    { embeddings },
  );

  const first = await service.search('What did we decide about job queues?', 1);
  assert.equal(first.results[0]?.entryId.value, 'queue-entry');
  assert.equal(first.results[0]?.sessionPath, sessionPath);
  assert.equal(first.indexSummary.embeddedChunks, 2);
  assert.equal(first.totalChunks, 2);

  const second = await service.search('queue decision', 1);
  assert.equal(second.indexSummary.embeddedChunks, 0);
  assert.deepEqual(embeddedInputs, [
    'We chose a durable queue for job delivery.',
    'The navigation bar is blue.',
    'What did we decide about job queues?',
    'queue decision',
  ]);
});
