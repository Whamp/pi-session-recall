import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { readSessionConversationChunks } from './session-conversation-index.js';

void test('session JSONL becomes searchable conversation chunks with provenance', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-session-'));
  const sessionPath = join(directory, 'session.jsonl');
  const oversized = `${'first '.repeat(30)}needle ${'last '.repeat(30)}`;
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
        id: 'user-1',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Where did we put the migration plan?' },
      },
      {
        type: 'message',
        id: 'assistant-1',
        parentId: 'user-1',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'secret reasoning' },
            { type: 'text', text: oversized },
            { type: 'toolCall', id: 'call-1', name: 'read', arguments: {} },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-1',
        parentId: 'assistant-1',
        timestamp: '2026-07-24T10:03:00Z',
        message: {
          role: 'toolResult',
          content: [{ type: 'text', text: 'giant private tool output' }],
        },
      },
      {
        type: 'compaction',
        id: 'compact-1',
        parentId: 'tool-1',
        timestamp: '2026-07-24T10:04:00Z',
        summary: 'The migration plan lives in docs/migration.md',
        tokensBefore: 1000,
      },
      {
        type: 'session_info',
        id: 'name-1',
        parentId: 'compact-1',
        timestamp: '2026-07-24T10:05:00Z',
        name: 'Migration planning',
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    maxCharacters: 120,
    overlapCharacters: 20,
  });

  assert.equal(chunks[0]?.sessionId.value, 'session-1');
  assert.equal(chunks[0]?.cwd, '/project');
  assert.equal(chunks[0]?.sessionName, 'Migration planning');
  assert.ok(
    chunks.some((chunk) => chunk.role === 'user' && chunk.content.includes('migration plan')),
  );
  assert.ok(chunks.some((chunk) => chunk.role === 'assistant' && chunk.content.includes('needle')));
  assert.ok(
    chunks.some((chunk) => chunk.role === 'summary' && chunk.entryId.value === 'compact-1'),
  );
  assert.ok(chunks.every((chunk) => chunk.content.length <= 120));
  assert.ok(chunks.every((chunk) => chunk.sessionPath === sessionPath));
  assert.ok(chunks.every((chunk) => !chunk.content.includes('secret reasoning')));
  assert.ok(chunks.every((chunk) => !chunk.content.includes('giant private tool output')));
  assert.equal(new Set(chunks.map((chunk) => chunk.id)).size, chunks.length);
});
