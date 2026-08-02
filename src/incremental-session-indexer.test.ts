import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

class MemoryConversationStore implements ConversationChunkStore {
  readonly chunks = new Map<string, IndexedSessionConversationChunk>();

  upsertChunks(chunks: IndexedSessionConversationChunk[]): void {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, structuredClone(chunk));
    }
  }

  deleteChunks(ids: string[]): void {
    for (const id of ids) {
      this.chunks.delete(id);
    }
  }

  fetchConversationChunks(ids: string[]) {
    return new Map(
      ids.flatMap((id) => {
        const chunk = this.chunks.get(id);
        return chunk ? [[id, structuredClone(chunk)]] : [];
      }),
    );
  }

  fetchVectors(ids: string[]) {
    return new Map(
      ids.flatMap((id) => {
        const chunk = this.chunks.get(id);
        return chunk?.isDenseSearchable ? [[id, [...chunk.embedding]]] : [];
      }),
    );
  }
}

type IndexingMaintenanceWorksetEvent = Extract<
  RecallIndexProgressEvent,
  { kind: 'indexing-maintenance-workset' }
>;

function isIndexingMaintenanceWorksetEvent(
  event: RecallIndexProgressEvent,
): event is IndexingMaintenanceWorksetEvent {
  return event.kind === 'indexing-maintenance-workset';
}

function sessionLines(entries: object[]): string {
  return `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`;
}

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createRecordingEmbeddingProvider(batches: string[][]): RecallEmbeddingProvider {
  return {
    async embedQuery(query) {
      return [query.length, 1, 0];
    },
    async embedDocuments(documents) {
      batches.push([...documents]);
      return documents.map((document) => [document.length, 1, 0]);
    },
  };
}

async function writeSimplePhysicalSessionFile(
  sessionPath: string,
  sessionId: string,
  content: string,
): Promise<void> {
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: sessionId, timestamp: '2026-01-01', cwd: '/p' },
      {
        type: 'message',
        id: `user-${sessionId}`,
        parentId: null,
        timestamp: '2026-01-01',
        message: { role: 'user', content },
      },
    ]),
  );
}

function createIndexerOptions(options: {
  sessionsDirectory: string;
  statePath: string;
  store: ConversationChunkStore;
  embeddingProvider: RecallEmbeddingProvider;
}) {
  return {
    ...options,
    tokenizer,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
}

void test('manual index maintenance announces planning and indexing before their work begins', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-phase-order-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const statePath = join(root, 'index-state.json');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'phase order evidence');
  const operationLog: string[] = [];
  const embeddingProvider = createRecordingEmbeddingProvider([]);

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      statePath,
      store: new MemoryConversationStore(),
      embeddingProvider: {
        ...embeddingProvider,
        async embedDocuments(documents, signal) {
          operationLog.push('embed documents');
          return embeddingProvider.embedDocuments(documents, signal);
        },
      },
    }),
    onProgress(event) {
      operationLog.push(`progress: ${event.kind}`);
    },
  });

  assert.deepEqual(operationLog.slice(0, 5), [
    'progress: discovering-physical-session-files',
    'progress: planning-maintenance-workset',
    'progress: maintenance-workset-planned',
    'progress: indexing-changed-physical-session-files',
    'embed documents',
  ]);
});

void test('manual index maintenance plans exact new, changed, unchanged, and missing files', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-workset-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const statePath = join(root, 'index-state.json');
  await mkdir(sessionsDirectory, { recursive: true });
  const changedPath = join(sessionsDirectory, 'changed.jsonl');
  const missingPath = join(sessionsDirectory, 'missing.jsonl');
  const unchangedPath = join(sessionsDirectory, 'unchanged.jsonl');
  await writeSimplePhysicalSessionFile(changedPath, 'changed', 'original changed evidence');
  await writeSimplePhysicalSessionFile(missingPath, 'missing', 'soon missing evidence');
  await writeSimplePhysicalSessionFile(unchangedPath, 'unchanged', 'stable evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);

  await writeSimplePhysicalSessionFile(
    changedPath,
    'changed',
    'updated and longer changed evidence',
  );
  await rm(missingPath);
  await writeSimplePhysicalSessionFile(join(sessionsDirectory, 'new.jsonl'), 'new', 'new evidence');
  const events: RecallIndexProgressEvent[] = [];

  const summary = await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      events.push(event);
    },
  });

  assert.equal(summary.scannedSessions, 3);
  assert.deepEqual(
    events.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 1,
      changedFiles: 1,
      missingFiles: 1,
      rebuild: false,
    },
  );

  const unchangedEvents: RecallIndexProgressEvent[] = [];
  await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      unchangedEvents.push(event);
    },
  });
  assert.deepEqual(
    unchangedEvents.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      rebuild: false,
    },
  );
});

void test('manual incremental indexing adds, reuses, changes, and deletes zvec rows', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const statePath = join(root, 'index-state.json');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  const entries: object[] = [
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
      message: { role: 'user', content: 'Remember the durable queue' },
    },
  ];
  await writeFile(sessionPath, sessionLines(entries));
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
  });

  const first = await indexChangedConversationSessions(options);
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.newlyEmbeddedChunks, 1);
  assert.equal(first.reusedVectors, 0);
  assert.equal(store.chunks.size, 1);

  const unchanged = await indexChangedConversationSessions(options);
  assert.equal(unchanged.indexedSessions, 0);
  assert.equal(unchanged.embeddingRequestCount, 0);
  assert.equal(embeddedBatches.length, 1);

  entries.push({
    type: 'message',
    id: 'assistant-1',
    parentId: 'user-1',
    timestamp: '2026-07-24T10:02:00Z',
    message: { role: 'assistant', content: 'Use one manually maintained zvec collection.' },
  });
  await writeFile(sessionPath, sessionLines(entries));
  const changed = await indexChangedConversationSessions(options);
  assert.equal(changed.indexedSessions, 1);
  assert.ok(changed.reusedVectors >= 1);
  assert.ok(changed.newlyEmbeddedChunks >= 1);
  assert.equal(embeddedBatches.length, 2);

  await rm(sessionPath);
  const deleted = await indexChangedConversationSessions(options);
  assert.equal(deleted.removedSessions, 1);
  assert.equal(store.chunks.size, 0);
});

void test('manual index maintenance reports cumulative progress and continues after a damaged file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const statePath = join(root, 'index-state.json');
  const damagedPath = join(sessionsDirectory, 'a-damaged.jsonl');
  const healthyPath = join(sessionsDirectory, 'z-healthy.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(damagedPath, 'damaged', 'initial damaged evidence');
  await writeSimplePhysicalSessionFile(healthyPath, 'healthy', 'initial healthy evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);

  await writeFile(damagedPath, '{"type":"session"}\n');
  await writeSimplePhysicalSessionFile(
    healthyPath,
    'healthy',
    'updated healthy evidence with more words',
  );
  const events: RecallIndexProgressEvent[] = [];
  const result = await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      events.push(event);
    },
  });

  const warningIndex = events.findIndex((event) => event.kind === 'physical-session-file-failed');
  const laterHealthyProgressIndex = events.findIndex(
    (event) =>
      event.kind === 'indexing-maintenance-workset' &&
      event.sessionPath === healthyPath &&
      event.completedFiles === 2,
  );
  assert.ok(warningIndex >= 0);
  assert.ok(laterHealthyProgressIndex > warningIndex);
  assert.deepEqual(events[warningIndex], {
    kind: 'physical-session-file-failed',
    sessionPath: damagedPath,
  });
  assert.equal(result.failedSessions.length, 1);
  assert.match(result.failedSessions[0]?.error ?? '', /session/iu);
  assert.ok([...store.chunks.values()].every((chunk) => chunk.sessionPath !== damagedPath));
  assert.ok([...store.chunks.values()].some((chunk) => chunk.sessionPath === healthyPath));

  const progressEvents = events.filter(isIndexingMaintenanceWorksetEvent);
  for (let index = 1; index < progressEvents.length; index += 1) {
    const previous = progressEvents[index - 1];
    const current = progressEvents[index];
    assert.ok(previous);
    assert.ok(current);
    assert.ok(current.completedFiles >= previous.completedFiles);
    assert.ok(current.indexedSessions >= previous.indexedSessions);
    assert.ok(current.newlyEmbeddedDocuments >= previous.newlyEmbeddedDocuments);
    assert.ok(current.reusedVectors >= previous.reusedVectors);
    assert.ok(current.deletedDocuments >= previous.deletedDocuments);
    assert.ok(current.failedSessions >= previous.failedSessions);
  }
  const finalProgress = progressEvents.at(-1);
  assert.ok(finalProgress);
  assert.equal(finalProgress.completedFiles, finalProgress.totalFiles);
  assert.equal(finalProgress.indexedSessions, result.indexedSessions);
  assert.equal(finalProgress.newlyEmbeddedDocuments, result.newlyEmbeddedChunks);
  assert.equal(finalProgress.reusedVectors, result.reusedVectors);
  assert.equal(finalProgress.deletedDocuments, result.deletedChunks);
  assert.equal(finalProgress.failedSessions, result.failedSessions.length);
});

void test('manual index maintenance reports multiple batches within one large physical session file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-batches-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const statePath = join(root, 'index-state.json');
  const sessionPath = join(sessionsDirectory, 'large.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  const messages = Array.from({ length: 140 }, (_, index) => ({
    type: 'message',
    id: `message-${index}`,
    parentId: index === 0 ? null : `message-${index - 1}`,
    timestamp: '2026-01-01',
    message: {
      role: index % 2 === 0 ? 'user' : 'assistant',
      content: `dense evidence message ${index}`,
    },
  }));
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: 'large', timestamp: '2026-01-01', cwd: '/p' },
      ...messages,
    ]),
  );
  const events: RecallIndexProgressEvent[] = [];

  const result = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      statePath,
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    onProgress(event) {
      events.push(event);
    },
  });

  const inFileProgress = events
    .filter(isIndexingMaintenanceWorksetEvent)
    .filter((event) => event.sessionPath === sessionPath && event.completedFiles === 0);
  assert.ok(inFileProgress.length >= 2);
  const firstBatchProgress = inFileProgress[0];
  const lastBatchProgress = inFileProgress.at(-1);
  assert.ok(firstBatchProgress);
  assert.ok(lastBatchProgress);
  assert.ok(lastBatchProgress.newlyEmbeddedDocuments > firstBatchProgress.newlyEmbeddedDocuments);
  assert.equal(result.embeddingRequestCount, inFileProgress.length);
  assert.ok(
    events.some(
      (event) =>
        event.kind === 'indexing-maintenance-workset' &&
        event.sessionPath === sessionPath &&
        event.completedFiles === 1,
    ),
  );
});

void test('manual incremental indexing keeps lexical-only tool evidence away from Octen', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-tool-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'tool.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: 'session-1', timestamp: '2026-01-01', cwd: '/p' },
      {
        type: 'message',
        id: 'assistant-1',
        parentId: null,
        timestamp: '2026-01-01',
        message: {
          role: 'assistant',
          content: [
            { type: 'toolCall', id: 'call-1', name: 'read', arguments: { path: '/tmp/a' } },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-1',
        parentId: 'assistant-1',
        timestamp: '2026-01-01',
        message: {
          role: 'toolResult',
          toolCallId: 'call-1',
          toolName: 'read',
          content: [{ type: 'text', text: 'source payload' }],
          isError: false,
        },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];

  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      statePath: join(root, 'index-state.json'),
      store,
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
  );

  assert.equal(embeddedBatches.length, 0);
  assert.ok([...store.chunks.values()].every((chunk) => !chunk.isDenseSearchable));
  assert.ok([...store.chunks.values()].some((chunk) => chunk.evidencePart === 'result'));
});
