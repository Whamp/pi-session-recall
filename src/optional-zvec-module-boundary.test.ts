import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

const NORMAL_V8_MODULE_PATHS = [
  'src/recall-extension.ts',
  'src/psr-cli.ts',
  'src/recall-conversation-service.ts',
  'src/recall-index-manifest.ts',
  'src/recall-conversation-config.ts',
  'src/session-source-search.ts',
  'src/incremental-session-indexer.ts',
  'src/certify-unified-sqlite-recall-production.ts',
] as const;

async function createZvecBlockingLoader(root: string): Promise<string> {
  const loaderPath = join(root, 'block-zvec-loader.mjs');
  await writeFile(
    loaderPath,
    `export async function resolve(specifier, context, nextResolve) {
  if (specifier === '@zvec/zvec') {
    throw new Error('Zvec Error: Failed to load prebuilt binary for darwin-x64. This platform may not be supported. Original error: missing optional binding');
  }
  return nextResolve(specifier, context);
}\n`,
  );
  return loaderPath;
}

function runWithZvecBlocked(loaderPath: string, source: string) {
  return spawnSync(
    process.execPath,
    [
      '--no-warnings',
      '--experimental-loader',
      pathToFileURL(loaderPath).href,
      '--import',
      'tsx',
      '--input-type=module',
      '--eval',
      source,
    ],
    { cwd: resolve('.'), encoding: 'utf8' },
  );
}

void test('normal v8 entry modules load when optional Zvec cannot be resolved', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-zvec-boundary-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const loaderPath = await createZvecBlockingLoader(root);
  const moduleUrls = NORMAL_V8_MODULE_PATHS.map(
    (modulePath) => pathToFileURL(resolve(modulePath)).href,
  );

  const result = runWithZvecBlocked(
    loaderPath,
    `await Promise.all(${JSON.stringify(moduleUrls)}.map((moduleUrl) => import(moduleUrl)));`,
  );

  assert.equal(result.status, 0, result.stderr);
});

void test('manifest-v6 dispatch explains when optional legacy rollback support is unavailable', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-zvec-unavailable-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const loaderPath = await createZvecBlockingLoader(root);
  const manifestPath = join(root, 'index-manifest.json');
  await writeFile(manifestPath, '{"manifestVersion":6}\n');
  const serviceUrl = pathToFileURL(resolve('src/recall-conversation-service.ts')).href;
  const config = {
    sessionsDirectory: join(root, 'sessions'),
    sqliteDatabasePath: join(root, 'recall.sqlite'),
    legacyV6ZvecDatabasePath: join(root, 'zvec'),
    legacyV6StatePath: join(root, 'index-state.json'),
    manifestPath,
    indexMaintenanceStatusPath: join(root, 'index-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(root, 'physical-session-ignore.json'),
    tokenizerCacheDirectory: join(root, 'tokenizer-cache'),
    lockPath: join(root, 'writer.lock'),
    embeddingBaseUrl: 'http://127.0.0.1:1',
    embeddingModel: 'test-model',
    embeddingServedModelId: 'test-model',
    embeddingNativeDimensions: 2,
    embeddingStoredDimensions: 2,
    embeddingBatchSize: 1,
    projectLineages: {},
    searchCandidateLimits: { dense: 10, invocation: 10 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
  const source = `
    const { createRecallConversationService } = await import(${JSON.stringify(serviceUrl)});
    const service = createRecallConversationService({
      ...${JSON.stringify(config)},
      projectLineages: new Map(),
    }, {
      embeddingProvider: { embedQuery: async () => [1, 0], embedDocuments: async () => [] },
    });
    try {
      await service.search('rollback evidence', 5, { scope: 'global' });
      throw new Error('Expected legacy-v6 rollback loading to fail');
    } catch (error) {
      if (!(error instanceof Error) || !/legacy-v6 rollback is unavailable[\\s\\S]*staged v8 generation is required/u.test(error.message)) {
        throw error;
      }
    }
  `;

  const result = runWithZvecBlocked(loaderPath, source);

  assert.equal(result.status, 0, result.stderr);
});

void test('@zvec/zvec remains pinned as an optional production dependency', async () => {
  const packageJson: unknown = JSON.parse(await readFile(resolve('package.json'), 'utf8'));
  assert.ok(packageJson && typeof packageJson === 'object');
  const optionalDependencies: unknown = Reflect.get(packageJson, 'optionalDependencies');
  const dependencies: unknown = Reflect.get(packageJson, 'dependencies');
  assert.deepEqual(optionalDependencies, { '@zvec/zvec': '0.6.0' });
  assert.ok(dependencies && typeof dependencies === 'object');
  assert.equal(Reflect.get(dependencies, '@zvec/zvec'), undefined);
});
