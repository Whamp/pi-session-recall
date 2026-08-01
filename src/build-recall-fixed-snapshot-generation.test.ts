import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecIndexType, ZVecMetricType, ZVecOpen } from '@zvec/zvec';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { RecallFixedSnapshotBuildFaultStage } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  createRecallGenerationComponentPaths,
  readRecallGenerationVectorValues,
} from './recall-generation-stores.js';
import { createStoredRecallEmbedding } from './recall-stored-embedding.js';

function createFloat32SensitiveNormalizedEmbedding(): number[] {
  const sensitiveComponent = 0.00024414061060257792;
  const repeatedComponent = Math.sqrt((1 - sensitiveComponent ** 2) / 1_022);
  const embedding = Array<number>(1_024).fill(0);
  for (let index = 0; index < 1_023; index += 1) {
    if (index !== 607) {
      embedding[index] = repeatedComponent;
    }
  }
  embedding[607] = sensitiveComponent;
  const norm = Math.hypot(...embedding);
  return embedding.map((value) => value / norm);
}

const BOOTSTRAP_INTERRUPTION_MODEL = [
  {
    stage: RecallFixedSnapshotBuildFaultStage.AFTER_GENERATION_DIRECTORY_CREATION,
    resumable: false,
  },
  { stage: RecallFixedSnapshotBuildFaultStage.AFTER_BOOTSTRAP_STATE_WRITE, resumable: false },
  { stage: RecallFixedSnapshotBuildFaultStage.AFTER_MANIFEST_WRITE, resumable: false },
  {
    stage: RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_DIRECTORY_CREATION,
    resumable: false,
  },
  {
    stage: RecallFixedSnapshotBuildFaultStage.AFTER_EXPECTED_SOURCE_DIRECTORY_CREATION,
    resumable: false,
  },
  { stage: RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_WRITE, resumable: false },
  { stage: RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_CAPTURE, resumable: true },
  {
    stage: RecallFixedSnapshotBuildFaultStage.AFTER_LEXICAL_SOURCE_STORE_CREATION,
    resumable: true,
  },
  { stage: RecallFixedSnapshotBuildFaultStage.AFTER_DENSE_STORE_CREATION, resumable: true },
  {
    stage: RecallFixedSnapshotBuildFaultStage.AFTER_SESSION_PROJECTION_STORE_CREATION,
    resumable: true,
  },
] as const;

void test('configured service builds one fixed snapshot into complete disposable generation stores', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'fixed.jsonl');
  await writeFile(
    sessionPath,
    `${JSON.stringify({ type: 'session', version: 3, id: 'fixed', timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'entry', parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: 'fixed snapshot ownership evidence' } })}\n`,
  );
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);

  const built = await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_fixed_snapshot_owner',
    physicalSessionPaths: [sessionPath],
  });

  assert.equal(built.generationId, 'generation_fixed_snapshot_owner');
  assert.ok(built.storeCounts.lexicalSource > 0);
  assert.ok(built.storeCounts.dense > 0);
  assert.ok(built.storeCounts.sessionProjection > 0);
  assert.ok(built.startingSnapshotFingerprint.length > 0);
});

void test('fixed snapshot build keeps one writable store session across physical sources', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-store-sessions-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const physicalSessionPaths = Array.from({ length: 129 }, (_, index) =>
    join(sessionsDirectory, `${String(index).padStart(3, '0')}.jsonl`),
  );
  await Promise.all(
    physicalSessionPaths.map((sessionPath, index) =>
      writeFile(
        sessionPath,
        `${JSON.stringify({ type: 'session', version: 3, id: `session-${index}`, timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: `entry-${index}`, parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: `bounded store session evidence ${index}` } })}\n`,
      ),
    ),
  );
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  let storeCloseCount = 0;
  const checkpointCounts: number[] = [];
  const durableEvents: string[] = [];
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    fixedSnapshotBuildFault(stage) {
      if (stage === RecallFixedSnapshotBuildFaultStage.AFTER_STORE_CLOSE) {
        storeCloseCount += 1;
        durableEvents.push('store-close');
      }
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_bounded_store_sessions',
    physicalSessionPaths,
    onPhysicalSourceCheckpoint(checkpoint) {
      checkpointCounts.push(checkpoint.completedPhysicalSourceCount);
      durableEvents.push('checkpoint');
    },
  });

  assert.equal(storeCloseCount, 1);
  assert.equal(durableEvents[0], 'store-close');
  assert.deepEqual(
    checkpointCounts,
    Array.from({ length: physicalSessionPaths.length }, (_, index) => index + 1),
  );
});

void test('fixed snapshot build bounds writable store sessions by generated record volume', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-record-sessions-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const largeSessionPath = join(sessionsDirectory, 'large.jsonl');
  const finalSessionPath = join(sessionsDirectory, 'final.jsonl');
  const reusedPhysicalSessionRecords = Array.from({ length: 600 }, (_, index) => [
    JSON.stringify({
      type: 'session',
      version: 3,
      id: `large-session-${index}`,
      timestamp: '2026-08-15T00:00:00.000Z',
      cwd: '/fixture',
    }),
    JSON.stringify({
      type: 'message',
      id: `large-entry-${index}`,
      parentId: null,
      timestamp: '2026-08-15T00:00:01.000Z',
      message: { role: 'assistant', content: `record volume evidence ${index}` },
    }),
  ]).flat();
  await Promise.all([
    writeFile(largeSessionPath, `${reusedPhysicalSessionRecords.join('\n')}\n`),
    writeFile(
      finalSessionPath,
      `${JSON.stringify({ type: 'session', version: 3, id: 'final-session', timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'final-entry', parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: 'final record volume evidence' } })}\n`,
    ),
  ]);
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  let storeCloseCount = 0;
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    fixedSnapshotBuildFault(stage) {
      if (stage === RecallFixedSnapshotBuildFaultStage.AFTER_STORE_CLOSE) {
        storeCloseCount += 1;
      }
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_record_bounded_store_sessions',
    physicalSessionPaths: [largeSessionPath, finalSessionPath],
  });

  assert.equal(storeCloseCount, 2);
});

void test('fixed snapshot build preserves normalized vector bytes in its inner-product index', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-vector-checksum-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'vector-checksum.jsonl');
  await writeFile(
    sessionPath,
    `${JSON.stringify({ type: 'session', version: 3, id: 'vector-checksum', timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'entry', parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: 'inner product vector checksum evidence' } })}\n`,
  );
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '1024',
    },
  });
  const embedding = createFloat32SensitiveNormalizedEmbedding();
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [...embedding];
      },
      async embedDocuments(documents) {
        return documents.map(() => [...embedding]);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  const built = await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_inner_product_vector',
    physicalSessionPaths: [sessionPath],
  });

  assert.deepEqual(await service.openValidatedRecallGeneration(built.generationId), built);
  const expectedStoredEmbedding = createStoredRecallEmbedding(embedding, {
    nativeDimensions: 1_024,
    storedDimensions: 1_024,
    source: 'inner product persistence expectation',
  });
  const expectedVectorChecksum = createHash('sha256')
    .update(Buffer.from(new Float32Array(expectedStoredEmbedding).buffer))
    .digest('hex');
  const densePath = createRecallGenerationComponentPaths(built.generationDirectory).denseStorePath;
  const dense = ZVecOpen(densePath, { readOnly: true });
  try {
    const vectorField = dense.schema.vectors().find(({ name }) => name === 'embedding');
    assert.equal(vectorField?.indexParams?.metricType, ZVecMetricType.IP);
    const [match] = dense.querySync({
      fieldName: 'embedding',
      vector: expectedStoredEmbedding,
      topk: 1,
      outputFields: ['vectorChecksum'],
      includeVector: true,
      params: { indexType: ZVecIndexType.HNSW, ef: 100 },
    });
    assert.ok(match);
    assert.equal(match.fields.vectorChecksum, expectedVectorChecksum);
    const persistedVectorChecksum = createHash('sha256')
      .update(
        Buffer.from(
          new Float32Array(readRecallGenerationVectorValues(match.vectors.embedding)).buffer,
        ),
      )
      .digest('hex');
    assert.equal(persistedVectorChecksum, expectedVectorChecksum);
  } finally {
    dense.closeSync();
  }
});

void test('configured service captures source bytes and identity through one opened descriptor', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-descriptor-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'descriptor.jsonl');
  const replacementPath = join(sessionsDirectory, 'replacement.jsonl');
  const createSource = (sessionId: string, entryId: string, content: string): string =>
    `${JSON.stringify({ type: 'session', version: 3, id: sessionId, timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: entryId, parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content } })}\n`;
  await Promise.all([
    writeFile(
      sessionPath,
      createSource('opened', 'opened-entry', 'OPENED_DESCRIPTOR_SENTINEL_122'),
    ),
    writeFile(
      replacementPath,
      createSource('replacement', 'replacement-entry', 'REPLACEMENT_PATHNAME_SENTINEL_122'),
    ),
  ]);
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  let sourcePathReplaced = false;
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    async fixedSnapshotBuildFault(stage) {
      if (stage === RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_OPEN) {
        await rename(replacementPath, sessionPath);
        sourcePathReplaced = true;
      }
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  const generation = await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_fixed_snapshot_descriptor',
    physicalSessionPaths: [sessionPath],
  });

  assert.equal(sourcePathReplaced, true);
  assert.ok(
    (
      await service.searchRecallGenerationLexical(
        generation.generationId,
        'OPENED_DESCRIPTOR_SENTINEL_122',
        10,
      )
    ).length > 0,
  );
  assert.deepEqual(
    await service.searchRecallGenerationLexical(
      generation.generationId,
      'REPLACEMENT_PATHNAME_SENTINEL_122',
      10,
    ),
    [],
  );
});

void test('replacement generation bootstrap interruption model resumes compatible states or discards safely', async (t) => {
  for (const transition of BOOTSTRAP_INTERRUPTION_MODEL) {
    await t.test(transition.stage, async (transitionTest) => {
      const root = await mkdtemp(join(tmpdir(), `recall-bootstrap-${transition.stage}-`));
      transitionTest.after(() => rm(root, { recursive: true, force: true }));
      const sessionsDirectory = join(root, 'sessions');
      const dataDirectory = join(root, 'data');
      await mkdir(sessionsDirectory, { recursive: true });
      const sessionPath = join(sessionsDirectory, 'bootstrap.jsonl');
      await writeFile(
        sessionPath,
        `${JSON.stringify({ type: 'session', version: 3, id: 'bootstrap', timestamp: '2026-08-16T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'entry', parentId: null, timestamp: '2026-08-16T00:00:01.000Z', message: { role: 'assistant', content: 'recoverable bootstrap evidence' } })}\n`,
      );
      const config = await loadRecallConversationConfig({
        environment: {
          PI_RECALL_DATA_DIRECTORY: dataDirectory,
          PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
          PI_RECALL_EMBEDDING_DIMENSIONS: '3',
        },
      });
      let interrupt = true;
      const service = createRecallConversationService(config, {
        embeddingProvider: {
          async embedQuery() {
            return [1, 0, 0];
          },
          async embedDocuments(documents) {
            return documents.map(() => [1, 0, 0]);
          },
        },
        async loadTokenizer() {
          return {
            encodeConversationText(text: string) {
              return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
            },
          };
        },
        fixedSnapshotBuildFault(stage) {
          if (stage === transition.stage && interrupt) {
            interrupt = false;
            throw new Error(`fixture bootstrap interruption: ${stage}`);
          }
        },
        workerSignal: { signalDetachedWorker() {} },
      });

      const generationId = `generation_${transition.stage.replaceAll('-', '_')}`;
      await assert.rejects(
        service.buildReplacementRecallGeneration({ generationId }),
        new RegExp(`fixture bootstrap interruption: ${transition.stage}`, 'u'),
      );
      const generationDirectory = join(config.generationRootDirectory, generationId);
      assert.equal(existsSync(generationDirectory), true);
      assert.equal(existsSync(config.activeGenerationPointerPath), false);
      assert.equal(
        existsSync(join(generationDirectory, 'build-bootstrap.json')),
        transition.stage !== RecallFixedSnapshotBuildFaultStage.AFTER_GENERATION_DIRECTORY_CREATION,
      );
      const snapshotDescriptorPath = join(generationDirectory, 'build-snapshot.json');
      const snapshotBeforeResume = existsSync(snapshotDescriptorPath)
        ? await readFile(snapshotDescriptorPath, 'utf8')
        : null;
      await writeFile(
        sessionPath,
        `${JSON.stringify({ type: 'session', version: 3, id: 'changed', timestamp: '2026-08-17T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'changed-entry', parentId: null, timestamp: '2026-08-17T00:00:01.000Z', message: { role: 'assistant', content: 'CHANGED_BOOTSTRAP_EVIDENCE' } })}\n`,
      );

      if (transition.resumable) {
        const resumed = await service.buildReplacementRecallGeneration({
          generationId,
          resumeExistingGeneration: true,
        });
        assert.equal(resumed.generationId, generationId);
        assert.deepEqual(await service.openValidatedRecallGeneration(generationId), resumed);
        assert.notEqual(snapshotBeforeResume, null);
        assert.equal(await readFile(snapshotDescriptorPath, 'utf8'), snapshotBeforeResume);
        assert.ok(
          (
            await service.searchRecallGenerationLexical(
              generationId,
              'recoverable bootstrap evidence',
              10,
            )
          ).length > 0,
        );
        assert.deepEqual(
          await service.searchRecallGenerationLexical(
            generationId,
            'CHANGED_BOOTSTRAP_EVIDENCE',
            10,
          ),
          [],
        );
      } else {
        await assert.rejects(
          service.buildReplacementRecallGeneration({
            generationId,
            resumeExistingGeneration: true,
          }),
          /Recall fixed snapshot generation (?:bootstrap state missing|bootstrap manifest missing|snapshot capture incomplete).*discard/u,
        );
      }

      assert.equal(existsSync(config.activeGenerationPointerPath), false);
      assert.equal(await service.discardStagingIndexGeneration(), true);
      assert.equal(await service.discardStagingIndexGeneration(), false);
      assert.equal(existsSync(generationDirectory), false);
    });
  }
});
