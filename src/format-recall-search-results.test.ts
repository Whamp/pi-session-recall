import assert from 'node:assert/strict';
import test from 'node:test';

import { formatRecallSearchResults } from './format-recall-search-results.js';

void test('recall results include concise excerpts and exact source provenance', () => {
  const output = formatRecallSearchResults(
    {
      totalChunks: 42,
      indexSummary: {
        scannedSessions: 10,
        indexedSessions: 1,
        removedSessions: 0,
        embeddedChunks: 2,
        deletedChunks: 0,
        failedSessions: [{ sessionPath: '/sessions/broken.jsonl', error: 'bad JSON' }],
      },
      results: [
        {
          id: 'chunk-1',
          checksum: 'sum-1',
          sessionId: { value: 'session-1' },
          sessionPath: '/sessions/one.jsonl',
          cwd: '/project',
          sessionName: 'Queue design',
          entryId: { value: 'entry-1' },
          role: 'assistant',
          timestamp: '2026-07-24T10:00:00Z',
          chunkIndex: 0,
          content: 'The durable queue decision and its tradeoffs are documented here.',
          score: 0.98765,
        },
      ],
    },
    40,
  );

  assert.match(output, /1\. Queue design/);
  assert.match(output, /2026-07-24T10:00:00Z · assistant · \/project/);
  assert.match(output, /Source: \/sessions\/one\.jsonl#entry-1/);
  assert.match(output, /0\.9877/);
  assert.match(output, /…/);
  assert.match(output, /Warning: 1 session failed to index/);
  assert.ok(!output.includes('checksum'));
});
