import assert from 'node:assert/strict';
import { join } from 'node:path';
import test from 'node:test';

import { SessionImportFormat } from './enums.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

void test('session JSONL importer preserves Unicode separators through the public document seam', async () => {
  const imported = await readSessionConversationImport(
    join(import.meta.dirname, 'fixtures/session-import/canonical-unicode-separators.jsonl'),
    { tokenizer },
  );

  assert.equal(imported.format, SessionImportFormat.CANONICAL_JSONL);
  assert.equal(imported.logicalSessions.length, 1);
  assert.deepEqual(
    imported.chunks.map((chunk) => chunk.content),
    ['literal\u2028line\u2029separators'],
  );
  assert.equal(imported.chunks[0]?.sourceLineStart, 2);
});
