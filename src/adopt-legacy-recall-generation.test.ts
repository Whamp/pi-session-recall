import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  adoptLegacyRecallGeneration,
  type AdoptLegacyRecallGenerationOptions,
} from './adopt-legacy-recall-generation.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  readRecallMaterialBacklogWarning,
} from './recall-generation-state.js';
import { createRecallIndexManifest, readRecallSearchManifest } from './recall-index-manifest.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import { rebuildRecallGeneration } from './rebuild-recall-generation.js';

void test('explicit adoption relocates exact version-5 layout read-only then retains it across version-6 cutover', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'adopt-legacy-recall-generation-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const dataDirectory = join(directory, 'recall');
  const sessionsDirectory = join(directory, 'sessions');
  const legacyDatabasePath = join(dataDirectory, 'zvec');
  const legacyStatePath = join(dataDirectory, 'index-state.json');
  const legacyManifestPath = join(dataDirectory, 'index-manifest.json');
  const generationRootDirectory = join(dataDirectory, 'generations');
  const activeGenerationPointerPath = join(dataDirectory, 'active-generation.json');
  const generationRegistryPath = join(dataDirectory, 'generation-registry.json');
  const backupEvidencePath = join(dataDirectory, 'legacy-adoption-backup.json');
  const backlogSummaryPath = join(dataDirectory, 'backlog-summary.json');
  const markerSpoolDirectory = join(dataDirectory, 'markers', 'pending');
  await mkdir(legacyDatabasePath, { recursive: true });
  await mkdir(sessionsDirectory);
  await mkdir(markerSpoolDirectory, { recursive: true });
  await writeFile(join(legacyDatabasePath, 'validated-zvec-sentinel'), 'legacy searchable data');
  const sessionSourcePath = join(sessionsDirectory, 'session.jsonl');
  await writeFile(sessionSourcePath, 'production-shaped source must remain untouched\n');
  await writeFile(legacyStatePath, '{"version":2,"importPolicyVersion":3,"sessions":{}}\n');
  const currentManifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: 'test-model',
      servedModelId: 'test/model',
      artifact: 'test.gguf',
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
    },
    canaryEmbedding: [1, 0, 0],
    projectLineages: normalizeRecallProjectLineages({}),
  });
  const {
    markerSchemaVersion,
    sessionProjectionSchemaVersion,
    manifestVersion,
    ...legacyManifestIdentity
  } = currentManifest;
  void markerSchemaVersion;
  void sessionProjectionSchemaVersion;
  void manifestVersion;
  await writeFile(
    legacyManifestPath,
    `${JSON.stringify({ ...legacyManifestIdentity, manifestVersion: 5 })}\n`,
  );
  let databaseValidationCount = 0;
  const adoptionOptions: AdoptLegacyRecallGenerationOptions = {
    dataDirectory,
    legacyDatabasePath,
    legacyStatePath,
    legacyManifestPath,
    generationRootDirectory,
    activeGenerationPointerPath,
    generationRegistryPath,
    backlogSummaryPath,
    backupEvidencePath,
    lockPath: join(dataDirectory, 'operation.lock'),
    nowEpochMilliseconds: () => 10_000,
    async validateLegacyDatabase(databasePath) {
      databaseValidationCount += 1;
      assert.equal(
        await readFile(join(databasePath, 'validated-zvec-sentinel'), 'utf8'),
        'legacy searchable data',
      );
    },
  };
  const adopted = await adoptLegacyRecallGeneration(adoptionOptions);

  assert.equal(databaseValidationCount, 2);
  assert.match(adopted.generationId, /^legacy-[a-f0-9]{24}$/u);
  await assert.rejects(() => access(legacyDatabasePath));
  await assert.rejects(() => access(legacyStatePath));
  await assert.rejects(() => access(legacyManifestPath));
  assert.equal(
    await readFile(join(adopted.generationDirectory, 'zvec', 'validated-zvec-sentinel'), 'utf8'),
    'legacy searchable data',
  );
  assert.equal(
    await readFile(sessionSourcePath, 'utf8'),
    'production-shaped source must remain untouched\n',
  );
  const adoptedSelection = await readRecallActiveGenerationSelection(
    activeGenerationPointerPath,
    generationRootDirectory,
  );
  assert.equal(adoptedSelection.activeGenerationId, adopted.generationId);
  assert.equal((await readRecallSearchManifest(adoptedSelection.manifestPath))?.manifestVersion, 6);
  const adoptedRegistry = await readRecallGenerationRegistry(generationRegistryPath);
  const legacyEntry = adoptedRegistry?.generations[0];
  assert.equal(legacyEntry?.state, RecallGenerationCutoverState.LEGACY_READ_ONLY);
  assert.equal(legacyEntry?.markerSchemaVersion, null);
  assert.equal(legacyEntry?.sessionProjectionSchemaVersion, null);
  assert.match(await readFile(backupEvidencePath, 'utf8'), /"state":"completed"/u);
  assert.match(
    (await readRecallMaterialBacklogWarning(backlogSummaryPath, adopted.generationId)) ?? '',
    /generationState=legacy_read_only/u,
  );

  await writeFile(
    backupEvidencePath,
    (await readFile(backupEvidencePath, 'utf8')).replace('"completed"', '"prepared"'),
  );
  await Promise.all([
    rm(activeGenerationPointerPath),
    rm(generationRegistryPath),
    rm(backlogSummaryPath),
  ]);
  const resumedAdoption = await adoptLegacyRecallGeneration(adoptionOptions);
  assert.deepEqual(resumedAdoption, adopted);
  assert.equal(databaseValidationCount, 3);
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.activeGenerationId,
    adopted.generationId,
  );

  await rebuildRecallGeneration({
    generationRootDirectory,
    activeGenerationPointerPath,
    generationRegistryPath,
    backlogSummaryPath,
    markerSpoolDirectory,
    lockPath: join(dataDirectory, 'operation.lock'),
    generationId: 'generation_schema_6',
    workerSignal: { signalDetachedWorker() {} },
    async buildGeneration(paths) {
      await mkdir(paths.databasePath, { recursive: true });
      await writeFile(paths.manifestPath, `${JSON.stringify(currentManifest)}\n`);
      return { result: null, async close() {} };
    },
    async validateGeneration() {
      return { indexManifestFingerprint: 'a'.repeat(64) };
    },
  });
  const rebuiltRegistry = await readRecallGenerationRegistry(generationRegistryPath);
  assert.equal(rebuiltRegistry?.activeGenerationId, 'generation_schema_6');
  assert.equal(rebuiltRegistry?.rollbackGenerationId, adopted.generationId);
  assert.equal(
    rebuiltRegistry?.generations.find(({ generationId }) => generationId === adopted.generationId)
      ?.state,
    RecallGenerationCutoverState.ROLLBACK,
  );
  await access(adopted.generationDirectory);
});
