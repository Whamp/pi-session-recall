import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { RecallSessionProjectionKind } from './enums.js';
import {
  createRecallGenerationValidationReceipt,
  writeRecallGenerationValidationReceipt,
} from './recall-generation-validation-receipt.js';
import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';

const ZVEC_MAXIMUM_WRITE_BATCH_SIZE = 1_024;
const COMPLETE_MEMBERSHIP_RECORD_COUNT = 119_663;

void test('configured service creates three independent real disposable zvec stores', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-stores-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const generationId = 'generation_stores_owner';
  await createRecallBackgroundIndexWorkerFixtureService(config).createEmptyRecallGeneration({
    generationId,
  });
  const generationDirectory = join(config.generationRootDirectory, generationId);
  const lexical = ZVecOpen(join(generationDirectory, 'lexical-source'), { readOnly: true });
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  const projections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(lexical.schema.vectors().length, 0);
    assert.equal(dense.schema.vectors().length, 1);
    assert.equal(projections.schema.vectors().length, 0);
    assert.notEqual(lexical.schema.name, dense.schema.name);
  } finally {
    lexical.closeSync();
    dense.closeSync();
    projections.closeSync();
  }
});

void test('configured service enumerates exact real-zvec generation membership beyond 119,662 rows', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-membership-limit-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_membership_limit';
  const generation = await service.createEmptyRecallGeneration({ generationId });
  await rm(generation.validationReceiptPath);

  const recordIds = Array.from(
    { length: COMPLETE_MEMBERSHIP_RECORD_COUNT },
    (_, ordinal) => `projection_logical_${String(ordinal).padStart(6, '0')}`,
  );
  const projectionStorePath = join(generation.generationDirectory, 'session-projections');
  const projectionStore = ZVecOpen(projectionStorePath);
  try {
    for (let start = 0; start < recordIds.length; start += ZVEC_MAXIMUM_WRITE_BATCH_SIZE) {
      const batchIds = recordIds.slice(start, start + ZVEC_MAXIMUM_WRITE_BATCH_SIZE);
      const statuses = projectionStore.insertSync(
        batchIds.map((id, offset) => {
          const ordinal = start + offset;
          return {
            id,
            fields: {
              schemaVersion: 1,
              generationId,
              projectionRecordId: id,
              projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
              physicalSourceIdentity: 'source_generated_fixture',
              logicalSessionOccurrenceId: `logical_${String(ordinal).padStart(6, '0')}`,
              projectionJson: '{}',
            },
          };
        }),
      );
      assert.equal(
        statuses.every(({ ok }) => ok),
        true,
      );
    }

    const readIds = (filter: string): string[] => {
      const ids: string[] = [];
      visitExactZvecDocuments(
        projectionStore,
        {
          filter,
          uniquePartitionField: 'logicalSessionOccurrenceId',
          outputFields: [],
        },
        ({ id }) => ids.push(id),
      );
      return ids.toSorted();
    };
    assert.deepEqual(readIds("logicalSessionOccurrenceId < 'logical_000000'"), []);
    assert.deepEqual(
      readIds("logicalSessionOccurrenceId < 'logical_000097'"),
      recordIds.slice(0, 97),
    );
    assert.deepEqual(
      readIds("logicalSessionOccurrenceId < 'logical_100000'"),
      recordIds.slice(0, 100_000),
    );
  } finally {
    projectionStore.closeSync();
  }

  await writeRecallGenerationValidationReceipt(
    generation.validationReceiptPath,
    createRecallGenerationValidationReceipt({
      generationId,
      manifestFingerprint: generation.manifestFingerprint,
      membership: {
        startingSnapshotFingerprint: generation.startingSnapshotFingerprint,
        physicalSourceCount: 0,
        logicalSessionOccurrenceCount: COMPLETE_MEMBERSHIP_RECORD_COUNT,
        lexicalSourceRecordIds: [],
        denseRecordIds: [],
        sessionProjectionRecordIds: recordIds,
      },
      validatedAtEpochMilliseconds: generation.validatedAtEpochMilliseconds,
    }),
  );

  const reopened = await service.openValidatedRecallGeneration(generationId);
  assert.equal(reopened.storeCounts.sessionProjection, COMPLETE_MEMBERSHIP_RECORD_COUNT);

  const resumed = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [],
    resumeExistingGeneration: true,
  });
  assert.equal(resumed.storeCounts.sessionProjection, COMPLETE_MEMBERSHIP_RECORD_COUNT);
});
