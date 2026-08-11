import assert from 'node:assert/strict';
import test from 'node:test';

import {
  installAutoIndexSchedule,
  uninstallAutoIndexSchedule,
  type AutoIndexSchedulerProcessResult,
  type AutoIndexSchedulerSystem,
} from './auto-index-scheduler.js';

interface AutoIndexSchedulerFixtureOptions {
  platform?: NodeJS.Platform;
  homeDirectory?: string;
  xdgConfigHome?: string;
  nodeExecutablePath?: string;
  packageRoot?: string;
  processResults?: AutoIndexSchedulerProcessResult[];
}

function createAutoIndexSchedulerFixture(options: AutoIndexSchedulerFixtureOptions = {}) {
  const files = new Map<string, string>();
  const fileModes = new Map<string, number>();
  const directories: string[] = [];
  const actions: string[] = [];
  const processCalls: Array<{ executable: string; argumentsList: readonly string[] }> = [];
  const system = {
    platform: options.platform ?? 'linux',
    homeDirectory: options.homeDirectory ?? '/home/recall-user',
    xdgConfigHome: options.xdgConfigHome ?? '/home/recall-user/.config-test',
    nodeExecutablePath: options.nodeExecutablePath ?? '/opt/node/bin/node',
    packageRoot: options.packageRoot ?? '/opt/pi session recall',
    async makeDirectory(directoryPath: string) {
      directories.push(directoryPath);
      actions.push(`mkdir ${directoryPath}`);
    },
    async writeFile(filePath: string, contents: string) {
      files.set(filePath, contents);
      actions.push(`write ${filePath}`);
    },
    async setFileMode(filePath: string, mode: number) {
      fileModes.set(filePath, mode);
      actions.push(`chmod ${mode.toString(8)} ${filePath}`);
    },
    async removeFile(filePath: string) {
      files.delete(filePath);
      actions.push(`remove ${filePath}`);
    },
    async runProcess(executable: string, argumentsList: readonly string[]) {
      processCalls.push({ executable, argumentsList });
      actions.push(`${executable} ${argumentsList.join(' ')}`);
      return options.processResults?.shift() ?? { exitCode: 0, stderr: '' };
    },
  } satisfies AutoIndexSchedulerSystem;
  return { system, files, fileModes, directories, actions, processCalls };
}

void test('installs a default hourly systemd user timer with direct absolute invocation', async () => {
  const fixture = createAutoIndexSchedulerFixture();
  const userUnitDirectory = '/home/recall-user/.config-test/systemd/user';
  fixture.files.set(`${userUnitDirectory}/pi-session-recall-optimize.service`, 'legacy service');
  fixture.files.set(`${userUnitDirectory}/pi-session-recall-optimize.timer`, 'legacy timer');

  const result = await installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system);

  assert.deepEqual(result, {});
  assert.equal(
    fixture.files.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.service',
    ),
    [
      '[Unit]',
      'Description=Maintain the pi-session-recall index',
      '',
      '[Service]',
      'Type=oneshot',
      'WorkingDirectory=/opt/pi session recall',
      'ExecStart="/opt/node/bin/node" --import tsx "/opt/pi session recall/bin/psr" index',
      'StandardOutput=journal',
      'StandardError=journal',
      '',
    ].join('\n'),
  );
  assert.equal(
    fixture.files.get('/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer'),
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
  const systemdService = [...fixture.files.entries()].find(([filePath]) =>
    filePath.endsWith('.service'),
  )?.[1];
  const systemdTimer = [...fixture.files.entries()].find(([filePath]) =>
    filePath.endsWith('.timer'),
  )?.[1];
  assert.doesNotMatch(systemdService ?? '', /Restart|RemainAfterExit/iu);
  assert.doesNotMatch(systemdTimer ?? '', /Persistent|OnCalendar/iu);
  assert.equal(fixture.files.size, 2);
  assert.ok(
    !fixture.files.has(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-optimize.timer',
    ),
  );
  assert.deepEqual(fixture.processCalls, [
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'disable', '--now', 'pi-session-recall-optimize.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'stop', 'pi-session-recall-optimize.service'],
    },
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
});

void test('uses the home systemd config directory when XDG_CONFIG_HOME is empty', async () => {
  const fixture = createAutoIndexSchedulerFixture({ xdgConfigHome: '' });

  await installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system);

  assert.ok(
    fixture.files.has('/home/recall-user/.config/systemd/user/pi-session-recall-index.timer'),
  );
});

void test('reinstall replaces the systemd definition and refreshes its interval', async () => {
  const fixture = createAutoIndexSchedulerFixture();

  await installAutoIndexSchedule({ value: 5n, unit: 'm' }, fixture.system);
  await installAutoIndexSchedule({ value: 2n, unit: 'h' }, fixture.system);

  assert.equal(fixture.files.size, 2);
  const timer =
    fixture.files.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer',
    ) ?? '';
  assert.match(timer, /OnActiveSec=2h\nOnUnitActiveSec=2h/u);
  assert.doesNotMatch(timer, /5min/u);
  assert.equal(fixture.processCalls.length, 12);
});

void test('fails installation when durable Linux timer setup fails', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    processResults: [
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 1, stderr: 'Failed to connect to user bus' },
    ],
  });

  await assert.rejects(
    installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system),
    /Auto-index scheduler command failed: systemctl --user daemon-reload: Failed to connect to user bus/u,
  );

  assert.equal(fixture.processCalls.length, 3);
});

void test('keeps the Linux timer installed when immediate indexing fails', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    processResults: [
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 1, stderr: 'psr index exited with status 1' },
    ],
  });

  const result = await installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system);

  assert.equal(fixture.files.size, 2);
  assert.equal(
    result.immediateRunWarning,
    'the immediate psr index attempt failed: psr index exited with status 1',
  );
});

void test('escapes systemd paths and renders explicit minutes', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    nodeExecutablePath: '/opt/node%build/bin/node',
    packageRoot: '/opt/pi$session %42',
  });

  await installAutoIndexSchedule({ value: 30n, unit: 'm' }, fixture.system);

  const service =
    fixture.files.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.service',
    ) ?? '';
  const timer =
    fixture.files.get(
      '/home/recall-user/.config-test/systemd/user/pi-session-recall-index.timer',
    ) ?? '';
  assert.match(service, /WorkingDirectory=\/opt\/pi\$session %%42/u);
  assert.match(
    service,
    /ExecStart="\/opt\/node%%build\/bin\/node" --import tsx "\/opt\/pi\$\$session %%42\/bin\/psr" index$/mu,
  );
  assert.match(timer, /OnActiveSec=30min\nOnUnitActiveSec=30min/u);
});

void test('installs a per-user macOS LaunchAgent with escaped paths and mode 0600', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    homeDirectory: '/Users/recall-user',
    nodeExecutablePath: '/Applications/Node & Tools/bin/node',
    packageRoot: '/Applications/Pi & Recall',
    processResults: [
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 0, stderr: '' },
    ],
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';
  const optimizePlistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-optimize.plist';

  fixture.files.set(optimizePlistPath, 'legacy optimization job');

  await installAutoIndexSchedule({ value: 15n, unit: 'm' }, fixture.system);

  const plist = fixture.files.get(plistPath) ?? '';
  assert.match(plist, /<string>\/Applications\/Node &amp; Tools\/bin\/node<\/string>/u);
  assert.match(plist, /<string>\/Applications\/Pi &amp; Recall<\/string>/u);
  assert.doesNotMatch(plist, /<string>--no-optimize<\/string>/u);
  assert.match(plist, /<integer>900<\/integer>/u);
  assert.match(plist, /<key>RunAtLoad<\/key>\n  <true\/>/u);
  assert.match(plist, /pi-session-recall-auto-index\.out\.log/u);
  assert.match(plist, /pi-session-recall-auto-index\.err\.log/u);
  assert.doesNotMatch(plist, /KeepAlive|LaunchOnlyOnce|Persistent|Restart/iu);
  assert.equal(fixture.files.has(optimizePlistPath), false);
  assert.equal(fixture.fileModes.get(plistPath), 0o600);
  assert.deepEqual(fixture.actions, [
    `launchctl unload ${plistPath}`,
    `launchctl unload ${optimizePlistPath}`,
    `remove ${optimizePlistPath}`,
    'mkdir /Users/recall-user/Library/LaunchAgents',
    'mkdir /Users/recall-user/.pi/agent/logs',
    `write ${plistPath}`,
    `chmod 600 ${plistPath}`,
    `launchctl load ${plistPath}`,
  ]);
});

void test('converts a macOS hour interval to seconds', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    homeDirectory: '/Users/recall-user',
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';

  await installAutoIndexSchedule({ value: 2n, unit: 'h' }, fixture.system);

  assert.match(fixture.files.get(plistPath) ?? '', /<integer>7200<\/integer>/u);
});

void test('fails macOS installation when launchctl cannot load the durable plist', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    homeDirectory: '/Users/recall-user',
    processResults: [
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 5, stderr: 'Load failed: invalid property list' },
    ],
  });

  await assert.rejects(
    installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system),
    /Auto-index scheduler command failed: launchctl load .*: Load failed: invalid property list/u,
  );

  assert.equal(fixture.files.size, 1);
});

void test('tolerates an already-unloaded macOS LaunchAgent during uninstall', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    homeDirectory: '/Users/recall-user',
    processResults: [
      { exitCode: 1, stderr: 'Could not find specified service' },
      { exitCode: 1, stderr: 'Could not find specified service' },
    ],
  });
  const plistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist';
  const optimizePlistPath =
    '/Users/recall-user/Library/LaunchAgents/dev.pi-session-recall.auto-optimize.plist';

  await uninstallAutoIndexSchedule(fixture.system);

  assert.deepEqual(fixture.actions, [
    `launchctl unload ${plistPath}`,
    `launchctl unload ${optimizePlistPath}`,
    `remove ${plistPath}`,
    `remove ${optimizePlistPath}`,
  ]);
});

void test('tolerates absent systemd user units during uninstall', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    processResults: [
      {
        exitCode: 1,
        stderr: 'Failed to disable unit: Unit pi-session-recall-index.timer does not exist',
      },
      {
        exitCode: 1,
        stderr: 'Failed to disable unit: Unit pi-session-recall-optimize.timer does not exist',
      },
      {
        exitCode: 5,
        stderr:
          'Failed to stop pi-session-recall-index.service: Unit pi-session-recall-index.service not loaded.',
      },
      {
        exitCode: 5,
        stderr:
          'Failed to stop pi-session-recall-optimize.service: Unit pi-session-recall-optimize.service not loaded.',
      },
      { exitCode: 0, stderr: '' },
    ],
  });

  await uninstallAutoIndexSchedule(fixture.system);

  assert.deepEqual(fixture.processCalls, [
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'disable', '--now', 'pi-session-recall-index.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'disable', '--now', 'pi-session-recall-optimize.timer'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'stop', 'pi-session-recall-index.service'],
    },
    {
      executable: 'systemctl',
      argumentsList: ['--user', 'stop', 'pi-session-recall-optimize.service'],
    },
    { executable: 'systemctl', argumentsList: ['--user', 'daemon-reload'] },
  ]);
});

void test('propagates an unexpected launchctl unload failure during install', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    processResults: [{ exitCode: 1, stderr: 'Unload failed: permission denied' }],
  });

  await assert.rejects(
    installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system),
    /Auto-index scheduler command failed: launchctl unload .*: Unload failed: permission denied/u,
  );
  assert.equal(fixture.files.size, 0);
});

void test('propagates an unexpected launchctl unload failure during uninstall', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    platform: 'darwin',
    processResults: [{ exitCode: 1, stderr: 'Unload failed: permission denied' }],
  });

  await assert.rejects(
    uninstallAutoIndexSchedule(fixture.system),
    /Auto-index scheduler command failed: launchctl unload .*: Unload failed: permission denied/u,
  );
  assert.equal(fixture.actions.length, 1);
});

void test('propagates a systemd timer disable failure during uninstall', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    processResults: [{ exitCode: 1, stderr: 'Failed to connect to user bus' }],
  });

  await assert.rejects(
    uninstallAutoIndexSchedule(fixture.system),
    /Auto-index scheduler command failed: systemctl --user disable --now pi-session-recall-index.timer: Failed to connect to user bus/u,
  );
  assert.equal(fixture.actions.length, 1);
});

void test('propagates a systemd service stop failure during uninstall', async () => {
  const fixture = createAutoIndexSchedulerFixture({
    processResults: [
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
      { exitCode: 1, stderr: 'Failed to connect to user bus' },
    ],
  });

  await assert.rejects(
    uninstallAutoIndexSchedule(fixture.system),
    /Auto-index scheduler command failed: systemctl --user stop pi-session-recall-index.service: Failed to connect to user bus/u,
  );
  assert.equal(fixture.actions.length, 3);
});

void test('rejects unsupported scheduler platforms', async () => {
  const fixture = createAutoIndexSchedulerFixture({ platform: 'win32' });

  await assert.rejects(
    installAutoIndexSchedule({ value: 1n, unit: 'h' }, fixture.system),
    /Auto-index scheduler is not supported on platform win32/u,
  );

  assert.equal(fixture.processCalls.length, 0);
});
