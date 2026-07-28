import assert from 'node:assert/strict';
import { mkdtemp, mkdir, open, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallBacklogFailureCategory, RecallGenerationCutoverState } from './enums.js';
import { RecallGenerationPointerError } from './errors.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import {
  calculateRecallActiveGenerationPointerChecksum,
  createRecallActiveGenerationPointer,
  decodeRecallActiveGenerationPointer,
  decodeRecallBacklogSummary,
  decodeRecallGenerationRegistry,
  encodeRecallActiveGenerationPointer,
  encodeRecallBacklogSummary,
  encodeRecallGenerationRegistry,
  readRecallActiveGenerationSelection,
  readRecallMaterialBacklogWarning,
  writeRecallActiveGenerationPointer,
  type RecallGenerationStateFilesystem,
  RECALL_ACTIVE_GENERATION_POINTER_VERSION,
  RECALL_BACKLOG_SUMMARY_VERSION,
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallBacklogSummary,
  type RecallGenerationRegistry,
} from './recall-generation-state.js';

const activeGenerationId = 'generation_2026_07_24';

function createRegistry(): RecallGenerationRegistry {
  const pointer = createRecallActiveGenerationPointer(activeGenerationId);
  return {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId,
    buildingGenerationId: 'generation_2026_07_25',
    rollbackGenerationId: 'generation_2026_07_23',
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: activeGenerationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1_753_315_200_000,
        stateChangedAtEpochMilliseconds: 1_753_318_800_000,
        rebuildStartMarkerId: 'marker_start',
      },
      {
        generationId: 'generation_2026_07_25',
        state: RecallGenerationCutoverState.BUILDING,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'b'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1_753_401_600_000,
        stateChangedAtEpochMilliseconds: 1_753_401_600_000,
        rebuildStartMarkerId: 'marker_next',
      },
      {
        generationId: 'generation_2026_07_23',
        state: RecallGenerationCutoverState.ROLLBACK,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'c'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1_753_228_800_000,
        stateChangedAtEpochMilliseconds: 1_753_318_800_000,
        rebuildStartMarkerId: null,
      },
    ],
  };
}

void test('active generation pointer is deterministic, checksummed, and strict', () => {
  const pointer = createRecallActiveGenerationPointer(activeGenerationId);
  assert.equal(pointer.version, RECALL_ACTIVE_GENERATION_POINTER_VERSION);
  assert.equal(pointer.checksum, calculateRecallActiveGenerationPointerChecksum(pointer));
  assert.match(pointer.checksum, /^[a-f0-9]{64}$/u);
  assert.deepEqual(
    decodeRecallActiveGenerationPointer(encodeRecallActiveGenerationPointer(pointer)),
    pointer,
  );
  assert.equal(createRecallActiveGenerationPointer(activeGenerationId).checksum, pointer.checksum);

  assert.throws(
    () => decodeRecallActiveGenerationPointer(JSON.stringify({ ...pointer, version: 2 })),
    /active generation pointer|invalid/iu,
  );
  assert.throws(
    () =>
      decodeRecallActiveGenerationPointer(JSON.stringify({ ...pointer, checksum: '0'.repeat(64) })),
    /checksum mismatch/iu,
  );
  assert.throws(
    () => decodeRecallActiveGenerationPointer(JSON.stringify({ ...pointer, unexpected: true })),
    /active generation pointer|invalid/iu,
  );
});

void test('generation registry round-trips cutover states and rejects inconsistent pointers', () => {
  const registry = createRegistry();
  assert.deepEqual(
    decodeRecallGenerationRegistry(encodeRecallGenerationRegistry(registry)),
    registry,
  );

  assert.throws(
    () =>
      encodeRecallGenerationRegistry({
        ...registry,
        activePointerChecksum: '0'.repeat(64),
      }),
    /pointer checksum mismatch/iu,
  );
  assert.throws(
    () =>
      encodeRecallGenerationRegistry({
        ...registry,
        activeGenerationId: 'generation_missing',
      }),
    /active generation|registry/iu,
  );
  assert.throws(
    () =>
      decodeRecallGenerationRegistry(
        JSON.stringify({
          ...registry,
          generations: [...registry.generations, registry.generations[0]],
        }),
      ),
    /duplicate generation/iu,
  );
  assert.throws(
    () => decodeRecallGenerationRegistry(JSON.stringify({ ...registry, version: 2 })),
    /generation registry|invalid/iu,
  );
  assert.throws(
    () =>
      decodeRecallGenerationRegistry(
        JSON.stringify({
          ...registry,
          generations: registry.generations.map((generation, index) =>
            index === 0 ? { ...generation, sessionProjectionSchemaVersion: 3 } : generation,
          ),
        }),
      ),
    /generation registry|invalid/iu,
  );
});

void test('search generation selection reads only one checksummed pointer directory', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-generation-selection-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const generationDirectory = join(generationRootDirectory, activeGenerationId);
  const pointerPath = join(directory, 'active-generation.json');
  await mkdir(generationDirectory, { recursive: true });
  await writeFile(
    pointerPath,
    encodeRecallActiveGenerationPointer(createRecallActiveGenerationPointer(activeGenerationId)),
  );

  assert.deepEqual(
    await readRecallActiveGenerationSelection(pointerPath, generationRootDirectory),
    {
      activeGenerationId,
      generationDirectory,
      databasePath: join(generationDirectory, 'zvec'),
      projectionDatabasePath: join(generationDirectory, 'session-projections'),
      statePath: join(generationDirectory, 'index-state.json'),
      manifestPath: join(generationDirectory, 'index-manifest.json'),
    },
  );

  await writeFile(
    pointerPath,
    encodeRecallActiveGenerationPointer(createRecallActiveGenerationPointer('generation_missing')),
  );
  await assert.rejects(
    () => readRecallActiveGenerationSelection(pointerPath, generationRootDirectory),
    RecallGenerationPointerError,
  );
  await writeFile(pointerPath, '{"version":1,"activeGenerationId":"other","checksum":"bad"}\n');
  await assert.rejects(
    () => readRecallActiveGenerationSelection(pointerPath, generationRootDirectory),
    RecallGenerationPointerError,
  );
  await rm(pointerPath);
  await assert.rejects(
    () => readRecallActiveGenerationSelection(pointerPath, generationRootDirectory),
    RecallGenerationPointerError,
  );
});

void test('generation selection rejects traversal and symlink escape without heuristic fallback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-generation-boundary-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const outsideDirectory = join(directory, 'outside');
  const pointerPath = join(directory, 'active-generation.json');
  await mkdir(generationRootDirectory);
  await mkdir(outsideDirectory);
  await symlink(outsideDirectory, join(generationRootDirectory, 'escaped_generation'));
  await writeFile(
    pointerPath,
    encodeRecallActiveGenerationPointer(createRecallActiveGenerationPointer('escaped_generation')),
  );

  await assert.rejects(
    () => readRecallActiveGenerationSelection(pointerPath, generationRootDirectory),
    /symlink escapes/iu,
  );
  assert.throws(() => createRecallActiveGenerationPointer('../outside'), /generation ID invalid/iu);
});

function createFaultingGenerationStateFilesystem(
  failureStage: 'directory-sync' | 'file-sync' | 'rename',
): RecallGenerationStateFilesystem {
  return {
    async createDirectory(path) {
      await mkdir(path, { recursive: true });
    },
    async openExclusiveFile(path) {
      const handle = await open(path, 'wx', 0o600);
      return {
        async writeFile(content) {
          await handle.writeFile(content);
        },
        async sync() {
          if (failureStage === 'file-sync') {
            throw new Error('injected generation state file sync failure');
          }
          await handle.sync();
        },
        async close() {
          await handle.close();
        },
      };
    },
    async renameFile(temporaryPath, destinationPath) {
      if (failureStage === 'rename') {
        throw new Error('injected generation state rename failure');
      }
      await rename(temporaryPath, destinationPath);
    },
    async syncDirectory(path) {
      if (failureStage === 'directory-sync') {
        throw new Error('injected generation state directory sync failure');
      }
      const handle = await open(path, 'r');
      try {
        await handle.sync();
      } finally {
        await handle.close();
      }
    },
    async removeFile(path) {
      await rm(path, { force: true });
    },
  };
}

void test('atomic pointer publication preserves the old pointer on fsync and rename failure', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-generation-atomic-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const pointerPath = join(directory, 'active-generation.json');
  const oldPointer = createRecallActiveGenerationPointer('generation_old');
  const newPointer = createRecallActiveGenerationPointer('generation_new');
  await writeRecallActiveGenerationPointer(pointerPath, oldPointer);

  for (const failureStage of ['file-sync', 'rename'] as const) {
    await assert.rejects(
      () =>
        writeRecallActiveGenerationPointer(pointerPath, newPointer, {
          filesystem: createFaultingGenerationStateFilesystem(failureStage),
        }),
      /injected generation state/iu,
    );
    assert.deepEqual(
      decodeRecallActiveGenerationPointer(await readFile(pointerPath, 'utf8')),
      oldPointer,
    );
  }
});

void test('directory fsync failure reports failure after exposing only the complete new pointer', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-generation-directory-sync-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const pointerPath = join(directory, 'active-generation.json');
  const oldPointer = createRecallActiveGenerationPointer('generation_old');
  const newPointer = createRecallActiveGenerationPointer('generation_new');
  await writeRecallActiveGenerationPointer(pointerPath, oldPointer);

  await assert.rejects(
    () =>
      writeRecallActiveGenerationPointer(pointerPath, newPointer, {
        filesystem: createFaultingGenerationStateFilesystem('directory-sync'),
      }),
    /directory sync failure/iu,
  );
  assert.deepEqual(
    decodeRecallActiveGenerationPointer(await readFile(pointerPath, 'utf8')),
    newPointer,
  );
});

void test('atomic pointer readers observe only the complete old or complete new pointer', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-generation-atomic-visibility-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const pointerPath = join(directory, 'active-generation.json');
  const oldPointer = createRecallActiveGenerationPointer('generation_old');
  const newPointer = createRecallActiveGenerationPointer('generation_new');
  await writeRecallActiveGenerationPointer(pointerPath, oldPointer);
  const observedGenerationIds = new Set<string>();
  const publication = writeRecallActiveGenerationPointer(pointerPath, newPointer);
  for (let index = 0; index < 100; index += 1) {
    const observed = decodeRecallActiveGenerationPointer(await readFile(pointerPath, 'utf8'));
    observedGenerationIds.add(observed.activeGenerationId);
  }
  await publication;
  observedGenerationIds.add(
    decodeRecallActiveGenerationPointer(await readFile(pointerPath, 'utf8')).activeGenerationId,
  );
  assert.ok(
    [...observedGenerationIds].every((id) => id === 'generation_old' || id === 'generation_new'),
  );
  assert.ok(observedGenerationIds.has('generation_new'));
});

void test('material backlog warning is scalar-only and silent for ordinary pending work', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-backlog-warning-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const backlogPath = join(directory, 'backlog-summary.json');
  const summary: RecallBacklogSummary = {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 2,
    oldestEligibleMarkerAgeMilliseconds: 60_000,
    activeGenerationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.ACTIVE,
    activeGenerationAgeMilliseconds: 86_400_000,
    rebuildAgeMilliseconds: null,
    lastFailureCategory: null,
    observedAtEpochMilliseconds: 1_753_401_700_000,
  };
  await writeFile(backlogPath, encodeRecallBacklogSummary(summary));
  assert.equal(await readRecallMaterialBacklogWarning(backlogPath, activeGenerationId), null);
  await writeFile(
    backlogPath,
    encodeRecallBacklogSummary({
      ...summary,
      oldestEligibleMarkerAgeMilliseconds: 30 * 60_000,
    }),
  );
  assert.equal(await readRecallMaterialBacklogWarning(backlogPath, activeGenerationId), null);

  const cases: Array<{
    name: string;
    summary: RecallBacklogSummary;
    expected: RegExp;
  }> = [
    {
      name: 'materially stale',
      summary: { ...summary, oldestEligibleMarkerAgeMilliseconds: 1_800_001 },
      expected: /oldestEligibleAgeMilliseconds=1800001/u,
    },
    {
      name: 'failed',
      summary: { ...summary, lastFailureCategory: RecallBacklogFailureCategory.WRITE_FAILED },
      expected: /lastFailureCategory=write_failed/u,
    },
    {
      name: 'rebuilding on older generation',
      summary: {
        ...summary,
        buildingGenerationId: 'generation_2026_07_25',
        generationState: RecallGenerationCutoverState.BUILDING,
      },
      expected: /generationState=building/u,
    },
  ];
  const serializedWarnings: string[] = [];
  for (const backlogCase of cases) {
    await writeFile(backlogPath, encodeRecallBacklogSummary(backlogCase.summary));
    const warning = await readRecallMaterialBacklogWarning(backlogPath, activeGenerationId);
    assert.match(warning ?? '', backlogCase.expected, backlogCase.name);
    serializedWarnings.push(warning ?? '');
  }
  const serialized = JSON.stringify(serializedWarnings);
  assert.doesNotMatch(serialized, /PRIVATE_CONVERSATION_SENTINEL|\/sessions\/private\.jsonl/u);
});

void test('backlog summary is scalar-only, versioned, strict, and contains no source details', () => {
  const summary: RecallBacklogSummary = {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 3,
    oldestEligibleMarkerAgeMilliseconds: 120_000,
    activeGenerationId,
    buildingGenerationId: 'generation_2026_07_25',
    generationState: RecallGenerationCutoverState.BUILDING,
    activeGenerationAgeMilliseconds: 86_400_000,
    rebuildAgeMilliseconds: 100_000,
    lastFailureCategory: RecallBacklogFailureCategory.WRITE_FAILED,
    observedAtEpochMilliseconds: 1_753_401_700_000,
  };
  const encoded = encodeRecallBacklogSummary(summary);
  assert.deepEqual(decodeRecallBacklogSummary(encoded), summary);
  assert.ok(
    Object.values(summary).every(
      (value) =>
        value === null ||
        typeof value === 'string' ||
        typeof value === 'number' ||
        typeof value === 'boolean',
    ),
  );

  const sentinel = 'PRIVATE_SOURCE_OR_ERROR_SENTINEL_49';
  assert.equal(encoded.includes(sentinel), false);
  assert.throws(
    () => decodeRecallBacklogSummary(JSON.stringify({ ...summary, sourcePath: sentinel })),
    /backlog summary|invalid/iu,
  );
  assert.throws(
    () => decodeRecallBacklogSummary(JSON.stringify({ ...summary, errorMessage: sentinel })),
    /backlog summary|invalid/iu,
  );
  assert.throws(
    () => decodeRecallBacklogSummary(JSON.stringify({ ...summary, version: 2 })),
    /backlog summary|invalid/iu,
  );
  assert.throws(
    () =>
      decodeRecallBacklogSummary(JSON.stringify({ ...summary, pendingEligibleSessionCount: -1 })),
    /backlog summary|invalid/iu,
  );
});
