import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { importSessionJsonl } from './import-session-jsonl.js';
import { parseRecallSessionGraph } from './parse-recall-session-record.js';
import {
  buildSessionConversationDocuments,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

void test('eligible document builder filters active-tail text before tokenization and waits for every turn contributor', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-eligible-documents-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    `${[
      {
        type: 'session',
        version: 3,
        id: 'logical',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'user',
        parentId: null,
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: 'old user evidence' },
      },
      {
        type: 'message',
        id: 'assistant',
        parentId: 'user',
        timestamp: '2026-01-01T00:00:02Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'old assistant evidence' }] },
      },
      {
        type: 'compaction',
        id: 'compact',
        parentId: 'assistant',
        timestamp: '2026-01-01T00:00:03Z',
        summary: 'immediate compact summary',
        firstKeptEntryId: 'assistant',
        tokensBefore: 100,
      },
      {
        type: 'branch_summary',
        id: 'branch',
        parentId: 'compact',
        timestamp: '2026-01-01T00:00:04Z',
        fromId: 'user',
        summary: 'immediate branch summary',
      },
      {
        type: 'message',
        id: 'tail',
        parentId: 'branch',
        timestamp: '2026-01-01T00:00:05Z',
        message: { role: 'user', content: 'ACTIVE TAIL MUST NOT TOKENIZE' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  const imported = await importSessionJsonl(sessionPath);
  const session = imported.sessions[0];
  assert.ok(session);
  const graph = parseRecallSessionGraph(session);
  const tokenizedTexts: string[] = [];
  const tokenizer: ConversationTextTokenizer = {
    encodeConversationText(text) {
      tokenizedTexts.push(text);
      return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
    },
  };

  const summaries = buildSessionConversationDocuments(graph, new Set(['compact', 'branch']), {
    sessionPath,
    logicalSessionIdentity: 'logical',
    tokenizer,
  });
  assert.deepEqual(summaries.map(({ entryId }) => entryId.value).toSorted(), ['branch', 'compact']);
  assert.ok(tokenizedTexts.every((text) => !text.includes('ACTIVE TAIL')));
  assert.ok(summaries.every(({ documentKind }) => documentKind === 'summary'));

  tokenizedTexts.length = 0;
  const incompleteTurn = buildSessionConversationDocuments(graph, new Set(['user']), {
    sessionPath,
    logicalSessionIdentity: 'logical',
    tokenizer,
  });
  assert.equal(
    incompleteTurn.some(({ documentKind }) => documentKind === 'turn_context'),
    false,
  );

  const completeTurn = buildSessionConversationDocuments(graph, new Set(['user', 'assistant']), {
    sessionPath,
    logicalSessionIdentity: 'logical',
    tokenizer,
  });
  assert.equal(
    completeTurn.some(({ documentKind }) => documentKind === 'turn_context'),
    true,
  );
  assert.ok(
    completeTurn.every(({ contributingEntryIds }) =>
      contributingEntryIds.every(({ value }) => value === 'user' || value === 'assistant'),
    ),
  );
  assert.ok(tokenizedTexts.every((text) => !text.includes('ACTIVE TAIL')));
});
