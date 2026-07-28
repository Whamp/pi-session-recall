import assert from 'node:assert/strict';
import { mkdtemp, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { SessionImportFormat } from './enums.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

const TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

async function snapshotSessionSource(sessionPath: string) {
  const [bytes, metadata] = await Promise.all([readFile(sessionPath), stat(sessionPath)]);
  return {
    bytes,
    size: metadata.size,
    mode: metadata.mode,
    mtimeMs: metadata.mtimeMs,
    ino: metadata.ino,
  };
}

void test('session JSONL importer preserves Unicode separators through the public document seam', async () => {
  const sessionPath = join(
    import.meta.dirname,
    'fixtures/session-import/canonical-unicode-separators.jsonl',
  );
  const imported = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });
  const repeatedImport = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });

  assert.equal(imported.format, SessionImportFormat.CANONICAL_JSONL);
  assert.deepEqual(
    repeatedImport.chunks.map(({ id }) => id),
    imported.chunks.map(({ id }) => id),
  );
  assert.equal(imported.logicalSessions.length, 1);
  assert.deepEqual(
    imported.chunks.map((chunk) => chunk.content),
    ['literal\u2028line\u2029separators'],
  );
  assert.deepEqual(imported.logicalSessions, [
    {
      sessionId: 'unicode-fixture',
      sourceLineStart: 1,
      sourceLineEnd: 2,
      entryIds: ['unicode-entry'],
      parentEntryIds: [null],
    },
  ]);
  assert.match(imported.chunks[0]?.id ?? '', /^[a-f0-9]{40}$/u);
  assert.deepEqual(
    imported.chunks.map((chunk) => ({
      checksum: chunk.checksum,
      entryId: chunk.entryId.value,
      parentEntryId: chunk.parentEntryId?.value ?? null,
      sourceLineStart: chunk.sourceLineStart,
      sourceLineEnd: chunk.sourceLineEnd,
      content: chunk.content,
    })),
    [
      {
        checksum: 'c1fe6fe86a3c831881eb9b7a9aa6b2477ad60188b8eee7269af62f1020fba97b',
        entryId: 'unicode-entry',
        parentEntryId: null,
        sourceLineStart: 2,
        sourceLineEnd: 2,
        content: 'literal\u2028line\u2029separators',
      },
    ],
  );
});

void test('session JSONL framing handles CR and LF split across stream chunks without source mutation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-crlf-chunk-seam-'));
  const sessionPath = join(directory, 'crlf-chunk-seam.jsonl');
  const header = {
    type: 'session',
    version: 3,
    id: 'crlf-chunk-seam',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
    padding: '',
  };
  const unpaddedHeader = JSON.stringify(header);
  header.padding = 'x'.repeat(65_535 - Buffer.byteLength(unpaddedHeader));
  const serializedHeader = JSON.stringify(header);
  assert.equal(Buffer.byteLength(serializedHeader), 65_535);
  const serializedMessage = JSON.stringify({
    type: 'message',
    id: 'entry',
    parentId: null,
    timestamp: '2026-07-24T10:01:00Z',
    message: { role: 'user', content: 'CRLF seam content' },
  });
  await writeFile(sessionPath, `${serializedHeader}\r\n${serializedMessage}\n`);
  const before = await snapshotSessionSource(sessionPath);

  const imported = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });

  assert.equal(imported.format, SessionImportFormat.CANONICAL_JSONL);
  assert.deepEqual(
    imported.chunks.map((chunk) => chunk.content),
    ['CRLF seam content'],
  );
  assert.deepEqual(await snapshotSessionSource(sessionPath), before);
});

void test('session JSONL framing removes only the CR before LF and keeps a final CR parseable', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-crlf-exactness-'));
  const sessionPath = join(directory, 'crlf-exactness.jsonl');
  const header = JSON.stringify({
    type: 'session',
    version: 3,
    id: 'crlf-exactness',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  });
  const message = JSON.stringify({
    type: 'message',
    id: 'entry',
    parentId: null,
    timestamp: '2026-07-24T10:01:00Z',
    message: { role: 'user', content: 'CRCRLF and final CR content' },
  });
  await writeFile(sessionPath, `${header}\r\r\n${message}\r`);
  const before = await snapshotSessionSource(sessionPath);

  const imported = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });

  assert.deepEqual(
    imported.chunks.map((chunk) => chunk.content),
    ['CRCRLF and final CR content'],
  );
  assert.equal(imported.chunks[0]?.sourceLineStart, 2);
  assert.deepEqual(await snapshotSessionSource(sessionPath), before);
});

void test('failed framing reads preserve source bytes and metadata', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-failed-framing-source-'));
  const sessionPath = join(directory, 'truncated.jsonl');
  await writeFile(
    sessionPath,
    '{"type":"session","version":3,"id":"truncated","timestamp":"2026-07-24T10:00:00Z","cwd":"/project"}\r\n{"type":"message","id":"unfinished',
  );
  const before = await snapshotSessionSource(sessionPath);

  await assert.rejects(
    () => readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER }),
    /Recall session JSON invalid.*truncated\.jsonl:2/u,
  );
  assert.deepEqual(await snapshotSessionSource(sessionPath), before);
});

void test('format routing keeps canonical lookalikes canonical and one header out of reuse history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-format-routing-guard-'));
  const sessionPath = join(directory, 'canonical-lookalike.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 2,
        id: 'canonical-lookalike',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'canonical-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'canonical despite v1-shaped message metadata' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  const imported = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });

  assert.equal(imported.format, SessionImportFormat.CANONICAL_JSONL);
  assert.equal(imported.logicalSessions.length, 1);
  assert.deepEqual(
    imported.chunks.map((chunk) => chunk.content),
    ['canonical despite v1-shaped message metadata'],
  );
});
