import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';

import { RecallEmbeddingProfile } from './enums.js';
import type {
  RecallConversationConfig,
  RecallConversationIndexOptions,
  RecallConversationMaintenanceService,
  RecallDatabaseTransition,
} from './recall-conversation-service.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { runPsrCli } from './psr-cli.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createPsrCliFixture(
  progressEvents: readonly RecallIndexProgressEvent[] = [],
  options: {
    monotonicTimes?: number[];
    failedSessions?: Array<{ sessionPath: string; error: string }>;
    fatalError?: Error;
    schedulerPlatform?: NodeJS.Platform;
    schedulerProcessResults?: Array<{ exitCode: number; stderr: string }>;
    physicalSessionIgnoreStatePath?: string;
    currentDirectory?: string;
    databaseTransition?: RecallDatabaseTransition;
    setupResult?: { exitCode: number; runInitialIndex: boolean; configPath?: string };
  } = {},
) {
  const calls: RecallConversationIndexOptions[] = [];
  const optimizeCalls: RecallConversationIndexOptions[] = [];
  const activateCalls: Array<{ databaseTarget: string; options: RecallConversationIndexOptions }> =
    [];
  const output: string[] = [];
  const progressOutput: string[] = [];
  const executionLog: string[] = [];
  const loadedConfigPaths: Array<string | undefined> = [];
  let closeCalls = 0;
  const schedulerProcessCalls: Array<{ executable: string; argumentsList: readonly string[] }> = [];
  const modelCommandCalls: string[][] = [];
  const setupCommandCalls: string[][] = [];
  const config: RecallConversationConfig = {
    sessionsDirectory: '/sessions',
    sqliteDatabasePath: '/recall/recall.sqlite',
    manifestPath: '/recall/index-manifest.json',
    indexMaintenanceStatusPath: '/recall/index-maintenance-status.json',
    physicalSessionIgnoreStatePath:
      options.physicalSessionIgnoreStatePath ?? '/recall/physical-session-ignore.json',
    tokenizerCacheDirectory: '/recall/tokenizers',
    lockPath: '/recall/operation.lock',
    embeddingProfile: RecallEmbeddingProfile.OCTEN_HTTP,
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-embed',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 2_560,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 16,
    localModelRootDirectory: '/recall-models',
    localEmbeddingParallelism: 4,
    localEmbeddingIntraOperationThreads: 4,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, invocation: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
  const service = {
    async close() {
      closeCalls += 1;
    },
    async search() {
      throw new Error('psr must not search');
    },
    async index(indexOptions) {
      calls.push(indexOptions ?? {});
      for (const event of progressEvents) {
        indexOptions?.onProgress?.(event);
      }
      if (options.fatalError) {
        throw options.fatalError;
      }
      return {
        totalChunks: 7,
        documentCounts: { dense: 7, invocations: 2 },
        databaseTransition: options.databaseTransition ?? { kind: 'active-updated' },
        indexSummary: {
          scannedSessions: 3,
          indexedSessions: 2,
          removedSessions: 1,
          reusedVectors: 4,
          newlyEmbeddedChunks: 5,
          embeddingRequestCount: 1,
          deletedChunks: 2,
          failedSessions: options.failedSessions ?? [],
        },
      };
    },
    async activate(databaseTarget, activateOptions) {
      activateCalls.push({ databaseTarget, options: activateOptions ?? {} });
      return { kind: 'staged-activated' as const };
    },
    async optimize(optimizeOptions) {
      optimizeCalls.push(optimizeOptions ?? {});
      optimizeOptions?.onProgress?.({ kind: 'optimizing-collection' });
      optimizeOptions?.onProgress?.({ kind: 'completed' });
      return { totalChunks: 7 };
    },
  } satisfies RecallConversationMaintenanceService;
  return {
    calls,
    optimizeCalls,
    activateCalls,
    output,
    progressOutput,
    executionLog,
    dependencies: {
      loadConfig: async (configPath?: string) => {
        executionLog.push('load config');
        loadedConfigPaths.push(configPath);
        return config;
      },
      createService(receivedConfig: RecallConversationConfig) {
        executionLog.push('create service');
        assert.equal(receivedConfig, config);
        return service;
      },
      writeOutput(text: string) {
        output.push(text);
      },
      writeProgress(text: string) {
        executionLog.push('write progress');
        progressOutput.push(text);
      },
      getMonotonicTimeMs() {
        return options.monotonicTimes?.shift() ?? 0;
      },
      getCurrentDirectory() {
        return options.currentDirectory ?? '/working-directory';
      },
      async runModelCommand(argumentsList: readonly string[]) {
        modelCommandCalls.push([...argumentsList]);
        return 0;
      },
      async runSetupCommand(argumentsList: readonly string[]) {
        setupCommandCalls.push([...argumentsList]);
        return options.setupResult ?? { exitCode: 0, runInitialIndex: false };
      },
      schedulerSystem: {
        platform: options.schedulerPlatform ?? 'linux',
        homeDirectory: '/home/recall-user',
        xdgConfigHome: '/home/recall-user/.config-test',
        nodeExecutablePath: '/opt/node/bin/node',
        packageRoot: '/opt/pi-session-recall',
        async makeDirectory() {},
        async writeFile() {},
        async setFileMode() {},
        async removeFile() {},
        async runProcess(executable: string, argumentsList: readonly string[]) {
          schedulerProcessCalls.push({ executable, argumentsList });
          return options.schedulerProcessResults?.shift() ?? { exitCode: 0, stderr: '' };
        },
      },
    },
    schedulerProcessCalls,
    modelCommandCalls,
    setupCommandCalls,
    loadedConfigPaths,
    get closeCalls() {
      return closeCalls;
    },
  };
}

void test('psr ignore add, list, and remove persist normalized exact paths without creating the service', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'psr-ignore-cli-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = createPsrCliFixture([], {
    physicalSessionIgnoreStatePath: join(root, 'physical-session-ignore.json'),
    currentDirectory: join(root, 'working'),
  });
  const normalizedPath = resolve(root, 'sessions', 'one.jsonl');

  assert.equal(
    await runPsrCli(['ignore', 'add', '../sessions/one.jsonl'], fixture.dependencies),
    0,
  );
  assert.equal(fixture.output.join(''), `Ignored: ${normalizedPath}\n`);
  assert.equal(
    await readFile(join(root, 'physical-session-ignore.json'), 'utf8'),
    `${JSON.stringify({ version: 1, ignoredPhysicalSessionPaths: [normalizedPath] })}\n`,
  );

  fixture.output.length = 0;
  assert.equal(await runPsrCli(['ignore', 'list'], fixture.dependencies), 0);
  assert.equal(fixture.output.join(''), `${normalizedPath}\n`);

  fixture.output.length = 0;
  assert.equal(
    await runPsrCli(['ignore', 'remove', '../sessions/one.jsonl'], fixture.dependencies),
    0,
  );
  assert.equal(fixture.output.join(''), `Removed: ${normalizedPath}\n`);
  fixture.output.length = 0;
  assert.equal(await runPsrCli(['ignore', 'list'], fixture.dependencies), 0);
  assert.equal(fixture.output.join(''), '');
  assert.ok(!fixture.executionLog.includes('create service'));
});

void test('psr ignore mutations are deterministic and idempotent for literal paths', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'psr-ignore-idempotent-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'physical-session-ignore.json');
  const fixture = createPsrCliFixture([], {
    physicalSessionIgnoreStatePath: statePath,
    currentDirectory: root,
  });
  const nonexistentPath = resolve(root, 'z-no-extension');
  const literalGlobPath = resolve(root, '*.jsonl');

  await runPsrCli(['ignore', 'add', 'z-no-extension'], fixture.dependencies);
  await runPsrCli(['ignore', 'add', '*.jsonl'], fixture.dependencies);
  const persisted = await readFile(statePath, 'utf8');
  const persistedInode = (await stat(statePath)).ino;
  fixture.output.length = 0;

  assert.equal(await runPsrCli(['ignore', 'add', 'z-no-extension'], fixture.dependencies), 0);
  assert.equal(fixture.output.join(''), `Already ignored: ${nonexistentPath}\n`);
  assert.equal(await readFile(statePath, 'utf8'), persisted);
  assert.equal((await stat(statePath)).ino, persistedInode);

  fixture.output.length = 0;
  assert.equal(await runPsrCli(['ignore', 'remove', 'absent'], fixture.dependencies), 0);
  assert.equal(fixture.output.join(''), `Not ignored: ${resolve(root, 'absent')}\n`);
  assert.equal(await readFile(statePath, 'utf8'), persisted);
  assert.equal((await stat(statePath)).ino, persistedInode);

  fixture.output.length = 0;
  await runPsrCli(['ignore', 'list'], fixture.dependencies);
  assert.equal(fixture.output.join(''), `${literalGlobPath}\n${nonexistentPath}\n`);
  assert.ok(!fixture.executionLog.includes('create service'));
});

void test('psr ignore rejects malformed or noncanonical persisted policy state', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'psr-ignore-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'physical-session-ignore.json');
  const fixture = createPsrCliFixture([], {
    physicalSessionIgnoreStatePath: statePath,
    currentDirectory: root,
  });
  const invalidStates = [
    '{',
    '{"version":2,"ignoredPhysicalSessionPaths":[]}',
    '{"version":1,"ignoredPhysicalSessionPaths":[],"extra":true}',
    '{"version":1,"ignoredPhysicalSessionPaths":["relative.jsonl"]}',
    '{"version":1,"ignoredPhysicalSessionPaths":["/same.jsonl","/same.jsonl"]}',
    '{"version":1,"ignoredPhysicalSessionPaths":["/z.jsonl","/a.jsonl"]}',
    '{"version":1,"ignoredPhysicalSessionPaths":["/not/../normalized.jsonl"]}',
  ];

  for (const invalidState of invalidStates) {
    await writeFile(statePath, `${invalidState}\n`, 'utf8');
    await assert.rejects(
      runPsrCli(['ignore', 'add', 'new.jsonl'], fixture.dependencies),
      new RegExp(`Physical session ignore state invalid at ${statePath}`, 'u'),
    );
    assert.equal(await readFile(statePath, 'utf8'), `${invalidState}\n`);
  }
  assert.ok(!fixture.executionLog.includes('create service'));
});

void test('psr ignore rejects invalid subcommands and arity with the complete usage', async () => {
  const fixture = createPsrCliFixture();
  const usage = [
    'psr usage: psr index [--rebuild] [--stage] [--resume] [--reuse-active-vectors] [--compact]',
    '           psr activate <database-target>',
    '           psr setup [--local|--external] [--yes] [--index] [profile options]',
    '           psr model status|download [--yes]|doctor',
    '           psr auto-index install [--interval <N>m|<N>h]',
    '           psr auto-index uninstall',
    '           psr ignore add <session-path>',
    '           psr ignore list',
    '           psr ignore remove <session-path>',
  ].join('\n');
  const invalidArguments = [
    ['ignore'],
    ['ignore', 'unknown'],
    ['ignore', 'add'],
    ['ignore', 'add', 'one', 'two'],
    ['ignore', 'list', 'one'],
    ['ignore', 'remove'],
    ['ignore', 'remove', 'one', 'two'],
  ];

  for (const argumentsList of invalidArguments) {
    await assert.rejects(runPsrCli(argumentsList, fixture.dependencies), (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(error.message, usage);
      return true;
    });
  }
  assert.deepEqual(fixture.executionLog, []);
});

void test('psr model delegates only model arguments without opening the recall service', async () => {
  const fixture = createPsrCliFixture();

  assert.equal(await runPsrCli(['model', 'download', '--yes'], fixture.dependencies), 0);
  assert.deepEqual(fixture.modelCommandCalls, [['download', '--yes']]);
  assert.deepEqual(fixture.calls, []);
  assert.ok(!fixture.executionLog.includes('load config'));
  assert.ok(!fixture.executionLog.includes('create service'));
});

void test('psr setup delegates profile arguments and optionally starts a rebuild', async () => {
  const configured = createPsrCliFixture();
  assert.equal(await runPsrCli(['setup', '--local', '--yes'], configured.dependencies), 0);
  assert.deepEqual(configured.setupCommandCalls, [['--local', '--yes']]);
  assert.deepEqual(configured.calls, []);

  const indexing = createPsrCliFixture([], {
    setupResult: {
      exitCode: 0,
      runInitialIndex: true,
      configPath: '/custom/recall.json',
    },
  });
  assert.equal(await runPsrCli(['setup', '--local', '--yes', '--index'], indexing.dependencies), 0);
  assert.deepEqual(indexing.setupCommandCalls, [['--local', '--yes', '--index']]);
  assert.equal(indexing.calls.length, 1);
  assert.equal(indexing.calls[0]?.rebuild, true);
  assert.deepEqual(indexing.loadedConfigPaths, ['/custom/recall.json']);
});

void test('psr index keeps progress on stderr and the completed summary on stdout', async () => {
  const fixture = createPsrCliFixture([
    { kind: 'discovering-physical-session-files' },
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 1,
      changedFiles: 1,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: false,
    },
    { kind: 'optimizing-collection' },
    { kind: 'completed' },
  ]);

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: false, optimize: false, onProgress: 'function' },
  );
  assert.equal(fixture.executionLog[0], 'write progress');
  assert.equal(fixture.executionLog[1], 'load config');
  assert.match(fixture.progressOutput.join(''), /Preparing/i);
  assert.match(fixture.progressOutput.join(''), /Discovering physical session files/i);
  assert.doesNotMatch(fixture.output.join(''), /Preparing|Discovering/iu);
  assert.match(fixture.output.join(''), /Sessions: 2 indexed of 3 scanned/iu);
  assert.match(fixture.output.join(''), /Dense documents: 7/iu);
  assert.match(fixture.output.join(''), /Compact Invocations: 2/iu);
  assert.doesNotMatch(fixture.progressOutput.join(''), /7 dense documents/iu);
});

void test('psr index --no-optimize remains an update-only compatibility flag', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['index', '--no-optimize'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: false, optimize: false, onProgress: 'function' },
  );
});

void test('psr rejects removed optimization commands and scheduler flags', async () => {
  const fixture = createPsrCliFixture();

  for (const argumentsList of [['optimize'], ['auto-index', 'install', '--optimize-daily']]) {
    await assert.rejects(runPsrCli(argumentsList, fixture.dependencies), /psr usage: psr index/u);
  }

  assert.deepEqual(fixture.calls, []);
  assert.deepEqual(fixture.optimizeCalls, []);
  assert.deepEqual(fixture.schedulerProcessCalls, []);
});

void test('psr index writes a readable multiline summary with elapsed time', async () => {
  const sessionPath = '/sessions/damaged.jsonl';
  const fixture = createPsrCliFixture([{ kind: 'completed' }], {
    monotonicTimes: [0, 65_000],
    failedSessions: [{ sessionPath, error: 'Session graph invalid: missing parent' }],
  });

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 1);
  assert.equal(
    fixture.output.join(''),
    [
      'Summary',
      '  Elapsed: 1m 05s',
      '  Sessions: 2 indexed of 3 scanned; 1 removed',
      '  Documents: 5 embedded; 4 vectors reused; 2 deleted',
      '  Dense documents: 7',
      '  Compact Invocations: 2',
      '  Failed sessions: 1',
      '',
      'Failures',
      `  ${sessionPath}`,
      '    Session graph invalid: missing parent',
      '',
    ].join('\n'),
  );
});

void test('psr index renders concise human progress with elapsed time and a rolling estimate', async () => {
  const sessionPath = '/sessions/a-very-long-physical-session-file-name.jsonl';
  const fixture = createPsrCliFixture(
    [
      { kind: 'indexing-changed-physical-session-files' },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 4,
        sessionPath,
        indexedSessions: 1,
        newlyEmbeddedDocuments: 1_234,
        reusedVectors: 56_789,
        deletedDocuments: 1,
        failedSessions: 0,
      },
      { kind: 'completed' },
    ],
    { monotonicTimes: [0, 1_000, 10_000, 65_000] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  const progress = fixture.progressOutput.join('');
  assert.match(progress, /Recall index maintenance/iu);
  assert.match(progress, /1\/4 files/iu);
  assert.match(progress, /10s elapsed/iu);
  assert.match(progress, /about 27s remaining/iu);
  assert.match(progress, /1,234 embedded/iu);
  assert.match(progress, /56,789 reused/iu);
  assert.match(progress, /Completed in 1m 05s/iu);
  assert.doesNotMatch(progress, new RegExp(sessionPath, 'u'));
});

void test('psr index waits for a healthy completed file before estimating remaining time', async () => {
  const fixture = createPsrCliFixture(
    [
      { kind: 'indexing-changed-physical-session-files' },
      { kind: 'physical-session-file-failed', sessionPath: '/sessions/damaged.jsonl' },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 4,
        sessionPath: '/sessions/damaged.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 0,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 1,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 2,
        totalFiles: 4,
        sessionPath: '/sessions/healthy.jsonl',
        indexedSessions: 1,
        newlyEmbeddedDocuments: 3,
        reusedVectors: 5,
        deletedDocuments: 0,
        failedSessions: 1,
      },
    ],
    { monotonicTimes: [0, 0, 100, 1_000, 11_000] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  const progressLines = fixture.progressOutput.filter((line) => /^\s+\d+\/\d+ files/u.test(line));
  assert.match(progressLines[0] ?? '', /estimating time remaining/iu);
  assert.match(progressLines[1] ?? '', /about 22s remaining/iu);
});

void test('psr index prints planning and indexing phase transitions in event order', async () => {
  const fixture = createPsrCliFixture([
    { kind: 'planning-maintenance-workset' },
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 1,
      newFiles: 1,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: false,
    },
    { kind: 'indexing-changed-physical-session-files' },
  ]);

  await runPsrCli(['index'], fixture.dependencies);

  assert.deepEqual(fixture.progressOutput.slice(1, 4), [
    'Planning maintenance workset...\n',
    [
      '',
      'Found 1 physical session file.',
      'Maintenance workset: 1 file (1 new, 0 changed, 0 missing, 0 ignored removals).',
      'Estimated time: calculating after the first file completes.',
      '',
    ].join('\n'),
    '\nIndexing maintenance workset...\n',
  ]);
});

void test('psr index describes empty and rebuild maintenance worksets in plain text', async () => {
  const emptyFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 4,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: false,
    },
  ]);
  await runPsrCli(['index'], emptyFixture.dependencies);
  assert.match(emptyFixture.progressOutput.join(''), /no files require indexing or removal/iu);

  const ignoredRemovalFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 1,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 1,
      rebuild: false,
    },
  ]);
  await runPsrCli(['index'], ignoredRemovalFixture.dependencies);
  assert.match(ignoredRemovalFixture.progressOutput.join(''), /0 missing, 1 ignored removal/iu);

  const rebuildFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 4,
      newFiles: 4,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: true,
    },
  ]);
  await runPsrCli(['index', '--rebuild'], rebuildFixture.dependencies);
  assert.match(rebuildFixture.progressOutput.join(''), /all 4 scheduled for indexing/iu);
  const rebuildProgress = rebuildFixture.progressOutput.join('');
  assert.ok(!rebuildProgress.includes('\u001b'));
  assert.ok(!rebuildProgress.includes('\r'));
  assert.ok(!rebuildProgress.includes('%'));
  assert.ok(rebuildFixture.progressOutput.every((line) => line.endsWith('\n')));
});

void test('psr index throttles routine updates but emits phase changes and warnings immediately', async () => {
  const failedPath = '/sessions/damaged.jsonl';
  const fixture = createPsrCliFixture(
    [
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 2,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 128,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 2,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 256,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      { kind: 'physical-session-file-failed', sessionPath: failedPath },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 2,
        sessionPath: failedPath,
        indexedSessions: 0,
        newlyEmbeddedDocuments: 256,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 2,
        sessionPath: '/sessions/two.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 300,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 2,
        totalFiles: 2,
        sessionPath: '/sessions/two.jsonl',
        indexedSessions: 1,
        newlyEmbeddedDocuments: 301,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      { kind: 'optimizing-collection' },
    ],
    { monotonicTimes: [0, 10, 500, 600, 700, 800, 1_700, 1_710] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  assert.equal(fixture.progressOutput.filter((line) => /^\s+\d+\/\d+ files/u.test(line)).length, 3);
  assert.match(fixture.progressOutput.join(''), new RegExp(failedPath, 'u'));
  assert.match(fixture.progressOutput.join(''), /Optimizing searchable collection/iu);
});

void test('psr index prints terminal file progress before the throttle interval elapses', async () => {
  const fixture = createPsrCliFixture(
    [
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 1,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 1,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 1,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 1,
        newlyEmbeddedDocuments: 1,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
    ],
    { monotonicTimes: [0, 0, 0] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  assert.deepEqual(
    fixture.progressOutput
      .filter((line) => /^\s+\d+\/\d+ files/u.test(line))
      .map((line) => line.match(/^\s+(\d+\/\d+) files/u)?.[1]),
    ['0/1', '1/1'],
  );
});

void test('psr index warns immediately while retaining complete failure details on stdout', async () => {
  const sessionPath = '/sessions/damaged.jsonl';
  const error = 'Session graph invalid: missing parent entry user-9';
  const fixture = createPsrCliFixture([{ kind: 'physical-session-file-failed', sessionPath }], {
    failedSessions: [{ sessionPath, error }],
  });

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 1);
  assert.match(fixture.progressOutput.join(''), new RegExp(sessionPath, 'u'));
  assert.doesNotMatch(fixture.progressOutput.join(''), new RegExp(error, 'u'));
  assert.match(fixture.output.join(''), new RegExp(sessionPath, 'u'));
  assert.match(fixture.output.join(''), new RegExp(error, 'u'));
});

void test('psr index releases embedding resources after a fatal service error', async () => {
  const fixture = createPsrCliFixture([], { fatalError: new Error('Store setup failed') });

  await assert.rejects(runPsrCli(['index'], fixture.dependencies), /Store setup failed/u);

  assert.equal(fixture.output.join(''), '');
  assert.match(fixture.progressOutput.join(''), /Preparing recall index/iu);
  assert.equal(fixture.closeCalls, 1);
});

void test('psr index --compact preserves the former one-line stdout summary', async () => {
  const sessionPath = '/sessions/damaged.jsonl';
  const error = 'Session graph invalid: missing parent';
  const fixture = createPsrCliFixture([], {
    failedSessions: [{ sessionPath, error }],
  });

  const exitCode = await runPsrCli(['index', '--compact'], fixture.dependencies);

  assert.equal(exitCode, 1);
  assert.equal(
    fixture.output.join(''),
    [
      'Indexed 2 of 3 sessions · removed 1 · embedded 5 · reused 4 vectors · deleted 2 documents · 7 dense documents · 2 compact Invocations · 1 failed sessions',
      `Failed: ${sessionPath}: ${error}`,
      '',
    ].join('\n'),
  );
});

void test('psr index --rebuild explicitly replaces the index', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['index', '--rebuild'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: true, optimize: false, onProgress: 'function' },
  );
});

void test('psr index --rebuild --stage leaves the database target ready for certification', async () => {
  const databaseTarget = 'generations/generation-certified';
  const fixture = createPsrCliFixture([{ kind: 'rebuild-candidate-staged', databaseTarget }], {
    databaseTransition: {
      kind: 'candidate-staged',
      databaseTarget,
      staleCandidatesRemoved: 0,
    },
  });

  const exitCode = await runPsrCli(['index', '--rebuild', '--stage'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: true, deferActivation: true, optimize: false, onProgress: 'function' },
  );
  assert.match(fixture.output.join(''), new RegExp(`staged at ${databaseTarget}`, 'u'));
  assert.match(fixture.output.join(''), /active database unchanged/iu);
});

void test('psr index resumes the interrupted staged rebuild without replacing its candidate', async () => {
  const databaseTarget = 'generations/generation-resumed';
  const fixture = createPsrCliFixture([{ kind: 'resuming-rebuild-candidate' }], {
    databaseTransition: {
      kind: 'candidate-staged',
      databaseTarget,
      staleCandidatesRemoved: 0,
    },
  });

  const exitCode = await runPsrCli(
    ['index', '--rebuild', '--stage', '--resume'],
    fixture.dependencies,
  );

  assert.equal(exitCode, 0);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    {
      rebuild: true,
      deferActivation: true,
      resumeCandidate: true,
      optimize: false,
      onProgress: 'function',
    },
  );
  assert.match(fixture.progressOutput.join(''), /Resuming the interrupted candidate/iu);
});

void test('psr index opts into checksum-verified active vector reuse for a staged rebuild', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(
    ['index', '--rebuild', '--stage', '--resume', '--reuse-active-vectors'],
    fixture.dependencies,
  );

  assert.equal(exitCode, 0);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    {
      rebuild: true,
      deferActivation: true,
      resumeCandidate: true,
      reuseActiveVectors: true,
      optimize: false,
      onProgress: 'function',
    },
  );
});

void test('psr activate switches only the named staged database target', async () => {
  const fixture = createPsrCliFixture();
  const databaseTarget = 'generations/generation-certified';

  const exitCode = await runPsrCli(['activate', databaseTarget], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.activateCalls.length, 1);
  assert.equal(fixture.activateCalls[0]?.databaseTarget, databaseTarget);
  assert.equal(fixture.output.join(''), 'Staged recall database activated.\n');
  assert.deepEqual(fixture.calls, []);
});

void test('psr index --stage requires an explicit rebuild', async () => {
  const fixture = createPsrCliFixture();

  await assert.rejects(
    runPsrCli(['index', '--stage'], fixture.dependencies),
    /psr index --stage requires --rebuild/u,
  );
  assert.deepEqual(fixture.calls, []);
});

void test('psr index --resume requires a staged rebuild', async () => {
  const fixture = createPsrCliFixture();

  for (const argumentsList of [
    ['index', '--resume'],
    ['index', '--rebuild', '--resume'],
    ['index', '--stage', '--resume'],
  ]) {
    await assert.rejects(
      runPsrCli(argumentsList, fixture.dependencies),
      /psr index --resume requires --rebuild --stage/u,
    );
  }
  assert.deepEqual(fixture.calls, []);
});

void test('psr index --reuse-active-vectors requires a staged rebuild', async () => {
  const fixture = createPsrCliFixture();

  for (const argumentsList of [
    ['index', '--reuse-active-vectors'],
    ['index', '--rebuild', '--reuse-active-vectors'],
    ['index', '--stage', '--reuse-active-vectors'],
  ]) {
    await assert.rejects(
      runPsrCli(argumentsList, fixture.dependencies),
      /psr index --reuse-active-vectors requires --rebuild --stage/u,
    );
  }
  assert.deepEqual(fixture.calls, []);
});

void test('psr rebuild output distinguishes activated, stale, and failed databases', async () => {
  const activated = createPsrCliFixture(
    [
      { kind: 'preparing-rebuild-candidate', staleCandidatesRemoved: 2 },
      { kind: 'rebuild-candidate-activated' },
    ],
    {
      databaseTransition: {
        kind: 'candidate-activated',
        staleCandidatesRemoved: 2,
      },
    },
  );
  assert.equal(await runPsrCli(['index', '--rebuild'], activated.dependencies), 0);
  assert.match(activated.progressOutput.join(''), /candidate recall database/iu);
  assert.match(activated.progressOutput.join(''), /2 stale candidate databases removed/iu);
  assert.match(activated.progressOutput.join(''), /candidate recall database activated/iu);
  assert.match(activated.output.join(''), /Dense documents: 7/iu);
  assert.match(activated.output.join(''), /Compact Invocations: 2/iu);
  assert.match(activated.output.join(''), /Database: activated/iu);

  const failed = createPsrCliFixture([{ kind: 'rebuild-candidate-failed' }], {
    failedSessions: [{ sessionPath: '/sessions/damaged.jsonl', error: 'damaged' }],
    databaseTransition: { kind: 'candidate-failed', staleCandidatesRemoved: 0 },
  });
  assert.equal(await runPsrCli(['index', '--rebuild'], failed.dependencies), 1);
  assert.match(failed.progressOutput.join(''), /candidate recall database failed/iu);
  assert.match(failed.output.join(''), /Dense documents: 7/iu);
  assert.match(failed.output.join(''), /Compact Invocations: 2/iu);
  assert.match(failed.output.join(''), /Database: candidate failed; active database unchanged/iu);
});

void test('psr auto-index install defaults to update-only indexing', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['auto-index', 'install'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 1h.\n');
  assert.equal(fixture.schedulerProcessCalls.length, 6);
  assert.deepEqual(fixture.calls, []);
});

void test('psr auto-index install accepts a custom update interval', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(
    ['auto-index', 'install', '--interval', '30m'],
    fixture.dependencies,
  );

  assert.equal(exitCode, 0);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 30m.\n');
  assert.equal(fixture.schedulerProcessCalls.length, 6);
});

void test('psr auto-index install reports a nonfatal immediate indexing warning', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerProcessResults: [
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 1, stderr: 'psr index exited with status 1' },
    ],
  });

  const exitCode = await runPsrCli(['auto-index', 'install'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 1h.\n');
  assert.match(
    fixture.progressOutput.join(''),
    /Warning: Automatic recall indexing was installed, but the immediate psr index attempt failed: psr index exited with status 1/iu,
  );
});

void test('psr auto-index uninstall routes scheduler removal and reports success', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['auto-index', 'uninstall'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing uninstalled.\n');
  assert.equal(fixture.schedulerProcessCalls.length, 5);
  assert.deepEqual(fixture.calls, []);
});

void test('psr auto-index rejects invalid intervals before touching index maintenance', async () => {
  const fixture = createPsrCliFixture();

  for (const interval of ['0m', '-1h', '1.5h', '1', '1H', ' 1h', '1d', '01h']) {
    await assert.rejects(
      runPsrCli(['auto-index', 'install', '--interval', interval], fixture.dependencies),
      /psr auto-index interval must be a positive whole number followed by m or h/u,
    );
  }

  assert.deepEqual(fixture.calls, []);
  assert.deepEqual(fixture.executionLog, []);
});

void test('psr auto-index rejects unsupported platforms with one clear error', async () => {
  const fixture = createPsrCliFixture([], { schedulerPlatform: 'win32' });

  await assert.rejects(
    runPsrCli(['auto-index', 'install'], fixture.dependencies),
    /Auto-index scheduler is not supported on platform win32/u,
  );

  assert.equal(fixture.schedulerProcessCalls.length, 0);
});

void test('psr rejects every command surface outside its documented commands', async () => {
  const fixture = createPsrCliFixture();

  await assert.rejects(
    runPsrCli(['resume'], fixture.dependencies),
    /psr usage: psr index \[--rebuild\]/,
  );
  assert.deepEqual(fixture.calls, []);
});
