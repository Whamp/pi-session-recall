import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  RecallConversationConfig,
  RecallConversationIndexOptions,
  RecallConversationService,
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
    schedulerHomeDirectory?: string;
    schedulerXdgConfigHome?: string;
    schedulerNodeExecutablePath?: string;
    schedulerPackageRoot?: string;
    schedulerProcessResults?: Array<{ exitCode: number; stderr: string }>;
  } = {},
) {
  const calls: RecallConversationIndexOptions[] = [];
  const output: string[] = [];
  const progressOutput: string[] = [];
  const executionLog: string[] = [];
  const schedulerFiles = new Map<string, string>();
  const schedulerFileModes = new Map<string, number>();
  const schedulerDirectories: string[] = [];
  const schedulerActions: string[] = [];
  const schedulerProcessCalls: Array<{ executable: string; argumentsList: readonly string[] }> = [];
  const config: RecallConversationConfig = {
    sessionsDirectory: '/sessions',
    databasePath: '/recall/zvec',
    statePath: '/recall/index-state.json',
    manifestPath: '/recall/index-manifest.json',
    tokenizerCacheDirectory: '/recall/tokenizers',
    lockPath: '/recall/operation.lock',
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-embed',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 2_560,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 16,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
  const service = {
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
  } satisfies RecallConversationService;
  return {
    calls,
    output,
    progressOutput,
    executionLog,
    dependencies: {
      loadConfig: async () => {
        executionLog.push('load config');
        return config;
      },
      createService(receivedConfig: RecallConversationConfig) {
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
      schedulerSystem: {
        platform: options.schedulerPlatform ?? 'linux',
        homeDirectory: options.schedulerHomeDirectory ?? '/home/recall-user',
        xdgConfigHome: options.schedulerXdgConfigHome ?? '/home/recall-user/.config-test',
        nodeExecutablePath: options.schedulerNodeExecutablePath ?? '/opt/node/bin/node',
        packageRoot: options.schedulerPackageRoot ?? '/opt/pi session recall',
        async makeDirectory(directoryPath: string) {
          schedulerDirectories.push(directoryPath);
          schedulerActions.push(`mkdir ${directoryPath}`);
        },
        async writeFile(filePath: string, contents: string) {
          schedulerFiles.set(filePath, contents);
          schedulerActions.push(`write ${filePath}`);
        },
        async setFileMode(filePath: string, mode: number) {
          schedulerFileModes.set(filePath, mode);
          schedulerActions.push(`chmod ${mode.toString(8)} ${filePath}`);
        },
        async removeFile(filePath: string) {
          schedulerFiles.delete(filePath);
          schedulerActions.push(`remove ${filePath}`);
        },
        async runProcess(executable: string, argumentsList: readonly string[]) {
          schedulerProcessCalls.push({ executable, argumentsList });
          schedulerActions.push(`${executable} ${argumentsList.join(' ')}`);
          return options.schedulerProcessResults?.shift() ?? { exitCode: 0, stderr: '' };
        },
      },
    },
    schedulerFiles,
    schedulerFileModes,
    schedulerDirectories,
    schedulerActions,
    schedulerProcessCalls,
  };
}

void test('psr index keeps progress on stderr and the completed summary on stdout', async () => {
  const fixture = createPsrCliFixture([
    { kind: 'discovering-physical-session-files' },
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 1,
      changedFiles: 1,
      missingFiles: 0,
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
    { rebuild: false, optimize: true, onProgress: 'function' },
  );
  assert.equal(fixture.executionLog[0], 'write progress');
  assert.equal(fixture.executionLog[1], 'load config');
  assert.match(fixture.progressOutput.join(''), /Preparing/i);
  assert.match(fixture.progressOutput.join(''), /Discovering physical session files/i);
  assert.doesNotMatch(fixture.output.join(''), /Preparing|Discovering/iu);
  assert.match(fixture.output.join(''), /Sessions: 2 indexed of 3 scanned/iu);
  assert.match(fixture.output.join(''), /Searchable documents: 7/iu);
  assert.doesNotMatch(fixture.progressOutput.join(''), /7 searchable documents/iu);
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
      '  Searchable documents: 7',
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
      'Maintenance workset: 1 file (1 new, 0 changed, 0 missing).',
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
      rebuild: false,
    },
  ]);
  await runPsrCli(['index'], emptyFixture.dependencies);
  assert.match(emptyFixture.progressOutput.join(''), /no files require indexing or removal/iu);

  const rebuildFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 4,
      newFiles: 4,
      changedFiles: 0,
      missingFiles: 0,
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

void test('psr index lets fatal service errors reach the standalone process handler', async () => {
  const fixture = createPsrCliFixture([], { fatalError: new Error('Store setup failed') });

  await assert.rejects(runPsrCli(['index'], fixture.dependencies), /Store setup failed/u);

  assert.equal(fixture.output.join(''), '');
  assert.match(fixture.progressOutput.join(''), /Preparing recall index/iu);
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
      'Indexed 2 of 3 sessions · removed 1 · embedded 5 · reused 4 vectors · deleted 2 documents · 7 searchable documents · 1 failed sessions',
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
    { rebuild: true, optimize: true, onProgress: 'function' },
  );
});

void test('psr auto-index install creates and starts a default hourly systemd user timer', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['auto-index', 'install'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(
    fixture.schedulerFiles.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.service',
    ),
    [
      '[Unit]',
      'Description=Maintain the pi-session-recall index',
      '',
      '[Service]',
      'Type=oneshot',
      'WorkingDirectory="/opt/pi session recall"',
      'ExecStart="/opt/node/bin/node" --import tsx "/opt/pi session recall/bin/psr" index',
      'StandardOutput=journal',
      'StandardError=journal',
      '',
    ].join('\n'),
  );
  assert.equal(
    fixture.schedulerFiles.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer',
    ),
    [
      '[Unit]',
      'Description=Schedule pi-session-recall index maintenance',
      '',
      '[Timer]',
      'OnActiveSec=1h',
      'OnUnitActiveSec=1h',
      '',
      '[Install]',
      'WantedBy=timers.target',
      '',
    ].join('\n'),
  );
  const systemdService = [...fixture.schedulerFiles.entries()].find(([filePath]) =>
    filePath.endsWith('.service'),
  )?.[1];
  const systemdTimer = [...fixture.schedulerFiles.entries()].find(([filePath]) =>
    filePath.endsWith('.timer'),
  )?.[1];
  assert.doesNotMatch(systemdService ?? '', /Restart|RemainAfterExit/iu);
  assert.doesNotMatch(systemdTimer ?? '', /Persistent|OnCalendar/iu);
  assert.deepEqual(fixture.schedulerProcessCalls, [
    { executable: 'systemctl', argumentsList: ['--user', 'daemon-reload'] },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'enable', 'pi-session-recall-index.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'restart', 'pi-session-recall-index.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'start', 'pi-session-recall-index.service'],
    },
  ]);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 1h.\n');
  assert.deepEqual(fixture.calls, []);
});

void test('psr auto-index uses the home config directory when XDG_CONFIG_HOME is empty', async () => {
  const fixture = createPsrCliFixture([], { schedulerXdgConfigHome: '' });

  await runPsrCli(['auto-index', 'install'], fixture.dependencies);

  assert.ok(
    fixture.schedulerFiles.has(
      '/home/recall-user/.config/systemd/user/pi-session-recall-index.timer',
    ),
  );
});

void test('psr auto-index reinstall replaces the systemd definition and refreshes its interval', async () => {
  const fixture = createPsrCliFixture();

  await runPsrCli(['auto-index', 'install', '--interval', '5m'], fixture.dependencies);
  await runPsrCli(['auto-index', 'install', '--interval', '2h'], fixture.dependencies);

  assert.equal(fixture.schedulerFiles.size, 2);
  const timer =
    fixture.schedulerFiles.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer',
    ) ?? '';
  assert.match(timer, /OnActiveSec=2h\nOnUnitActiveSec=2h/u);
  assert.doesNotMatch(timer, /5min/u);
  assert.equal(fixture.schedulerProcessCalls.length, 8);
});

void test('psr auto-index install fails when durable Linux timer setup fails', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerProcessResults: [{ exitCode: 1, stderr: 'Failed to connect to user bus' }],
  });

  await assert.rejects(
    runPsrCli(['auto-index', 'install'], fixture.dependencies),
    /Auto-index scheduler command failed: systemctl --user daemon-reload: Failed to connect to user bus/u,
  );

  assert.equal(fixture.schedulerProcessCalls.length, 1);
  assert.equal(fixture.output.join(''), '');
});

void test('psr auto-index install keeps the Linux timer after immediate indexing fails', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerProcessResults: [
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 1, stderr: 'psr index exited with status 1' },
    ],
  });

  const exitCode = await runPsrCli(['auto-index', 'install'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.schedulerFiles.size, 2);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 1h.\n');
  assert.match(
    fixture.progressOutput.join(''),
    /Warning: Automatic recall indexing was installed, but the immediate psr index attempt failed: psr index exited with status 1/iu,
  );
});

void test('psr auto-index install escapes systemd paths and renders explicit minutes', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerNodeExecutablePath: '/opt/node%build/bin/node',
    schedulerPackageRoot: '/opt/pi$session %42',
  });

  await runPsrCli(['auto-index', 'install', '--interval', '30m'], fixture.dependencies);

  const service =
    fixture.schedulerFiles.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.service',
    ) ?? '';
  const timer =
    fixture.schedulerFiles.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer',
    ) ?? '';
  assert.match(service, /WorkingDirectory="\/opt\/pi\$\$session %%42"/u);
  assert.match(
    service,
    /ExecStart="\/opt\/node%%build\/bin\/node" --import tsx "\/opt\/pi\$\$session %%42\/bin\/psr" index/u,
  );
  assert.match(timer, /OnActiveSec=30min\nOnUnitActiveSec=30min/u);
});

void test('psr auto-index install creates a per-user macOS LaunchAgent', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerPlatform: 'darwin',
    schedulerHomeDirectory: '/Users/recall-user',
    schedulerNodeExecutablePath: '/Applications/Node & Tools/bin/node',
    schedulerPackageRoot: '/Applications/Pi & Recall',
    schedulerProcessResults: [
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 0, stderr: '' },
    ],
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';

  const exitCode = await runPsrCli(
    ['auto-index', 'install', '--interval', '15m'],
    fixture.dependencies,
  );

  assert.equal(exitCode, 0);
  assert.equal(
    fixture.schedulerFiles.get(plistPath),
    [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
      '<plist version="1.0">',
      '<dict>',
      '  <key>Label</key>',
      '  <string>dev.pi-session-recall.auto-index</string>',
      '  <key>ProgramArguments</key>',
      '  <array>',
      '    <string>/Applications/Node &amp; Tools/bin/node</string>',
      '    <string>--import</string>',
      '    <string>tsx</string>',
      '    <string>/Applications/Pi &amp; Recall/bin/psr</string>',
      '    <string>index</string>',
      '  </array>',
      '  <key>WorkingDirectory</key>',
      '  <string>/Applications/Pi &amp; Recall</string>',
      '  <key>StartInterval</key>',
      '  <integer>900</integer>',
      '  <key>RunAtLoad</key>',
      '  <true/>',
      '  <key>StandardOutPath</key>',
      '  <string>/Users/recall-user/.pi/agent/logs/pi-session-recall-auto-index.out.log</string>',
      '  <key>StandardErrorPath</key>',
      '  <string>/Users/recall-user/.pi/agent/logs/pi-session-recall-auto-index.err.log</string>',
      '</dict>',
      '</plist>',
      '',
    ].join('\n'),
  );
  assert.equal(fixture.schedulerFileModes.get(plistPath), 0o600);
  assert.deepEqual(fixture.schedulerActions, [
    `launchctl unload ${plistPath}`,
    'mkdir /Users/recall-user/Library/LaunchAgents',
    'mkdir /Users/recall-user/.pi/agent/logs',
    `write ${plistPath}`,
    `chmod 600 ${plistPath}`,
    `launchctl load ${plistPath}`,
  ]);
  assert.deepEqual(fixture.schedulerDirectories, [
    '/Users/recall-user/Library/LaunchAgents',
    '/Users/recall-user/.pi/agent/logs',
  ]);
  const plist = fixture.schedulerFiles.get(plistPath) ?? '';
  assert.doesNotMatch(plist, /KeepAlive|LaunchOnlyOnce|Persistent|Restart/iu);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing installed every 15m.\n');
  assert.equal(fixture.progressOutput.join(''), '');
  assert.deepEqual(fixture.calls, []);
});

void test('psr auto-index converts an explicit macOS hour interval to seconds', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerPlatform: 'darwin',
    schedulerHomeDirectory: '/Users/recall-user',
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';

  await runPsrCli(['auto-index', 'install', '--interval', '2h'], fixture.dependencies);

  assert.match(fixture.schedulerFiles.get(plistPath) ?? '', /<integer>7200<\/integer>/u);
});

void test('psr auto-index install fails when launchctl cannot load the durable plist', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerPlatform: 'darwin',
    schedulerHomeDirectory: '/Users/recall-user',
    schedulerProcessResults: [
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 5, stderr: 'Load failed: invalid property list' },
    ],
  });

  await assert.rejects(
    runPsrCli(['auto-index', 'install'], fixture.dependencies),
    /Auto-index scheduler command failed: launchctl load .*: Load failed: invalid property list/u,
  );

  assert.equal(fixture.schedulerFiles.size, 1);
  assert.equal(fixture.output.join(''), '');
});

void test('psr auto-index uninstall safely removes an absent macOS LaunchAgent', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerPlatform: 'darwin',
    schedulerHomeDirectory: '/Users/recall-user',
    schedulerProcessResults: [{ exitCode: 1, stderr: 'Could not find specified service' }],
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';

  const exitCode = await runPsrCli(['auto-index', 'uninstall'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.deepEqual(fixture.schedulerActions, [
    `launchctl unload ${plistPath}`,
    `remove ${plistPath}`,
  ]);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing uninstalled.\n');
  assert.deepEqual(fixture.calls, []);
});

void test('psr auto-index uninstall safely removes an absent systemd user schedule', async () => {
  const fixture = createPsrCliFixture([], {
    schedulerProcessResults: [
      { exitCode: 1, stderr: 'Unit pi-session-recall-index.timer does not exist' },
      { exitCode: 5, stderr: 'Unit pi-session-recall-index.service not loaded' },
      { exitCode: 0, stderr: '' },
    ],
  });

  const exitCode = await runPsrCli(['auto-index', 'uninstall'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.deepEqual(fixture.schedulerProcessCalls, [
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'disable', '--now', 'pi-session-recall-index.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'stop', 'pi-session-recall-index.service'],
    },
    { executable: 'systemctl', argumentsList: ['--user', 'daemon-reload'] },
  ]);
  assert.equal(fixture.schedulerFiles.size, 0);
  assert.equal(fixture.output.join(''), 'Automatic recall indexing uninstalled.\n');
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

void test('psr rejects every command surface other than index and auto-index', async () => {
  const fixture = createPsrCliFixture();

  await assert.rejects(
    runPsrCli(['resume'], fixture.dependencies),
    /psr usage: psr index \[--rebuild\]/,
  );
  assert.deepEqual(fixture.calls, []);
});
