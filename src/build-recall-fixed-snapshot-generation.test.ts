import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { RecallFixedSnapshotBuildFaultStage } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';

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
