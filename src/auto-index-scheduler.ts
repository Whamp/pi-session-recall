import { spawn } from 'node:child_process';
import { chmod, mkdir, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SYSTEMD_SERVICE_NAME = 'pi-session-recall-index.service';
const SYSTEMD_TIMER_NAME = 'pi-session-recall-index.timer';
const SYSTEMD_OPTIMIZE_SERVICE_NAME = 'pi-session-recall-optimize.service';
const SYSTEMD_OPTIMIZE_TIMER_NAME = 'pi-session-recall-optimize.timer';
const LAUNCH_AGENT_LABEL = 'dev.pi-session-recall.auto-index';
const LAUNCH_AGENT_FILE_NAME = `${LAUNCH_AGENT_LABEL}.plist`;
const LAUNCH_AGENT_OPTIMIZE_LABEL = 'dev.pi-session-recall.auto-optimize';
const LAUNCH_AGENT_OPTIMIZE_FILE_NAME = `${LAUNCH_AGENT_OPTIMIZE_LABEL}.plist`;

/** A positive whole-number interval expressed in minutes or hours. */
export interface AutoIndexInterval {
  value: bigint;
  unit: 'm' | 'h';
}

/** Observable filesystem and process boundaries used by native per-user schedulers. */
export interface AutoIndexSchedulerSystem {
  platform: NodeJS.Platform;
  homeDirectory: string;
  xdgConfigHome?: string;
  nodeExecutablePath: string;
  packageRoot: string;
  makeDirectory: (directoryPath: string) => Promise<void>;
  writeFile: (filePath: string, contents: string) => Promise<void>;
  setFileMode: (filePath: string, mode: number) => Promise<void>;
  removeFile: (filePath: string) => Promise<void>;
  runProcess: (
    executable: string,
    argumentsList: readonly string[],
  ) => Promise<AutoIndexSchedulerProcessResult>;
}

/** The native scheduler process result, with stderr retained for clear CLI errors. */
export interface AutoIndexSchedulerProcessResult {
  exitCode: number;
  stderr: string;
}

/** Reports a nonfatal immediate indexing failure after durable scheduler installation. */
export interface AutoIndexInstallationResult {
  immediateRunWarning?: string;
}

async function runSchedulerProcess(
  executable: string,
  argumentsList: readonly string[],
): Promise<AutoIndexSchedulerProcessResult> {
  return await new Promise((resolveProcess, rejectProcess) => {
    const child = spawn(executable, argumentsList, {
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    let stderr = '';
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk;
    });
    child.on('error', rejectProcess);
    child.on('close', (exitCode) => {
      resolveProcess({ exitCode: exitCode ?? 1, stderr });
    });
  });
}

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const XDG_CONFIG_HOME = process.env.XDG_CONFIG_HOME;

/** Real current-user boundaries used by the installed standalone `psr` command. */
export const DEFAULT_AUTO_INDEX_SCHEDULER_SYSTEM: AutoIndexSchedulerSystem = {
  platform: process.platform,
  homeDirectory: homedir(),
  ...(XDG_CONFIG_HOME === undefined || XDG_CONFIG_HOME.length === 0
    ? {}
    : { xdgConfigHome: XDG_CONFIG_HOME }),
  nodeExecutablePath: resolve(process.execPath),
  packageRoot: resolve(PACKAGE_ROOT),
  async makeDirectory(directoryPath) {
    await mkdir(directoryPath, { recursive: true });
  },
  async writeFile(filePath, contents) {
    await writeFile(filePath, contents, 'utf8');
  },
  async setFileMode(filePath, mode) {
    await chmod(filePath, mode);
  },
  async removeFile(filePath) {
    await rm(filePath, { force: true });
  },
  runProcess: runSchedulerProcess,
};

function quoteSystemdValue(value: string): string {
  const escaped = value
    .replaceAll('\\', '\\\\')
    .replaceAll('"', '\\"')
    .replaceAll('$', () => '$$')
    .replaceAll('%', '%%');
  return `"${escaped}"`;
}

function escapeSystemdWorkingDirectory(value: string): string {
  return value.replaceAll('%', '%%');
}

function renderSystemdInterval(interval: AutoIndexInterval): string {
  return interval.unit === 'm' ? `${interval.value}min` : `${interval.value}h`;
}

function readSystemdConfigHome(system: AutoIndexSchedulerSystem): string {
  return system.xdgConfigHome === undefined || system.xdgConfigHome.length === 0
    ? join(system.homeDirectory, '.config')
    : system.xdgConfigHome;
}

function escapePlistXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

interface LaunchAgentSchedule {
  label: string;
  psrArguments: readonly string[];
  timingLines: readonly string[];
  runAtLoad: boolean;
  logName: string;
}

function renderLaunchAgentPlist(
  schedule: LaunchAgentSchedule,
  system: AutoIndexSchedulerSystem,
): string {
  const logsDirectory = join(system.homeDirectory, '.pi', 'agent', 'logs');
  const argumentsList = [
    system.nodeExecutablePath,
    '--import',
    'tsx',
    join(system.packageRoot, 'bin', 'psr'),
    ...schedule.psrArguments,
  ];
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    '  <key>Label</key>',
    `  <string>${schedule.label}</string>`,
    '  <key>ProgramArguments</key>',
    '  <array>',
    ...argumentsList.map((argument) => `    <string>${escapePlistXml(argument)}</string>`),
    '  </array>',
    '  <key>WorkingDirectory</key>',
    `  <string>${escapePlistXml(system.packageRoot)}</string>`,
    ...schedule.timingLines,
    ...(schedule.runAtLoad ? ['  <key>RunAtLoad</key>', '  <true/>'] : []),
    '  <key>StandardOutPath</key>',
    `  <string>${escapePlistXml(join(logsDirectory, `${schedule.logName}.out.log`))}</string>`,
    '  <key>StandardErrorPath</key>',
    `  <string>${escapePlistXml(join(logsDirectory, `${schedule.logName}.err.log`))}</string>`,
    '</dict>',
    '</plist>',
    '',
  ].join('\n');
}

function renderLaunchAgentIndexPlist(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): string {
  const intervalSeconds = interval.value * (interval.unit === 'm' ? 60n : 3_600n);
  return renderLaunchAgentPlist(
    {
      label: LAUNCH_AGENT_LABEL,
      psrArguments: ['index', '--no-optimize'],
      timingLines: ['  <key>StartInterval</key>', `  <integer>${intervalSeconds}</integer>`],
      runAtLoad: true,
      logName: 'pi-session-recall-auto-index',
    },
    system,
  );
}

function renderLaunchAgentOptimizePlist(system: AutoIndexSchedulerSystem): string {
  return renderLaunchAgentPlist(
    {
      label: LAUNCH_AGENT_OPTIMIZE_LABEL,
      psrArguments: ['optimize'],
      timingLines: [
        '  <key>StartCalendarInterval</key>',
        '  <dict>',
        '    <key>Hour</key>',
        '    <integer>23</integer>',
        '    <key>Minute</key>',
        '    <integer>0</integer>',
        '  </dict>',
      ],
      runAtLoad: false,
      logName: 'pi-session-recall-auto-optimize',
    },
    system,
  );
}

function renderSystemdService(
  system: AutoIndexSchedulerSystem,
  description: string,
  psrArguments: readonly string[],
): string {
  const executablePath = quoteSystemdValue(system.nodeExecutablePath);
  const packageRoot = escapeSystemdWorkingDirectory(system.packageRoot);
  const psrExecutablePath = quoteSystemdValue(join(system.packageRoot, 'bin', 'psr'));
  return [
    '[Unit]',
    `Description=${description}`,
    '',
    '[Service]',
    'Type=oneshot',
    `WorkingDirectory=${packageRoot}`,
    `ExecStart=${executablePath} --import tsx ${psrExecutablePath} ${psrArguments.join(' ')}`,
    'StandardOutput=journal',
    'StandardError=journal',
    '',
  ].join('\n');
}

function renderSystemdTimer(interval: AutoIndexInterval): string {
  const renderedInterval = renderSystemdInterval(interval);
  return [
    '[Unit]',
    'Description=Schedule pi-session-recall index maintenance',
    '',
    '[Timer]',
    `OnActiveSec=${renderedInterval}`,
    `OnUnitActiveSec=${renderedInterval}`,
    '',
    '[Install]',
    'WantedBy=timers.target',
    '',
  ].join('\n');
}

function renderSystemdOptimizeTimer(): string {
  return [
    '[Unit]',
    'Description=Schedule daily pi-session-recall optimization',
    '',
    '[Timer]',
    'OnCalendar=*-*-* 23:00:00',
    'Persistent=true',
    '',
    '[Install]',
    'WantedBy=timers.target',
    '',
  ].join('\n');
}

function throwSchedulerProcessError(
  executable: string,
  argumentsList: readonly string[],
  result: AutoIndexSchedulerProcessResult,
): never {
  const detail = result.stderr.trim();
  const suffix = detail.length === 0 ? '' : `: ${detail}`;
  throw new Error(
    `Auto-index scheduler command failed: ${executable} ${argumentsList.join(' ')}${suffix}`,
  );
}

async function runRequiredSchedulerProcess(
  system: AutoIndexSchedulerSystem,
  executable: string,
  argumentsList: readonly string[],
): Promise<void> {
  const result = await system.runProcess(executable, argumentsList);
  if (result.exitCode !== 0) {
    throwSchedulerProcessError(executable, argumentsList, result);
  }
}

function isLaunchAgentAlreadyUnloaded(result: AutoIndexSchedulerProcessResult): boolean {
  const detail = result.stderr.toLowerCase();
  return (
    detail.includes('could not find specified service') ||
    detail.includes('nothing found to unload') ||
    detail.includes('no such process')
  );
}

async function unloadLaunchAgent(
  system: AutoIndexSchedulerSystem,
  plistPath: string,
): Promise<void> {
  const argumentsList = ['unload', plistPath];
  const result = await system.runProcess('launchctl', argumentsList);
  if (result.exitCode !== 0 && !isLaunchAgentAlreadyUnloaded(result)) {
    throwSchedulerProcessError('launchctl', argumentsList, result);
  }
}

function isSystemdUnitAbsent(result: AutoIndexSchedulerProcessResult, unitName: string): boolean {
  const detail = result.stderr.toLowerCase();
  const normalizedUnitName = unitName.toLowerCase();
  return (
    detail.includes(`unit ${normalizedUnitName} does not exist`) ||
    detail.includes(`unit file ${normalizedUnitName} does not exist`) ||
    detail.includes(`unit ${normalizedUnitName} not loaded`)
  );
}

async function runSystemdUninstallProcess(
  system: AutoIndexSchedulerSystem,
  argumentsList: readonly string[],
  unitName: string,
): Promise<void> {
  const result = await system.runProcess('systemctl', argumentsList);
  if (result.exitCode !== 0 && !isSystemdUnitAbsent(result, unitName)) {
    throwSchedulerProcessError('systemctl', argumentsList, result);
  }
}

async function installSystemdAutoIndexSchedule(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): Promise<AutoIndexInstallationResult> {
  const userUnitDirectory = join(readSystemdConfigHome(system), 'systemd', 'user');
  const servicePath = join(userUnitDirectory, SYSTEMD_SERVICE_NAME);
  const timerPath = join(userUnitDirectory, SYSTEMD_TIMER_NAME);
  const optimizeServicePath = join(userUnitDirectory, SYSTEMD_OPTIMIZE_SERVICE_NAME);
  const optimizeTimerPath = join(userUnitDirectory, SYSTEMD_OPTIMIZE_TIMER_NAME);
  await system.makeDirectory(userUnitDirectory);
  await system.writeFile(
    servicePath,
    renderSystemdService(system, 'Maintain the pi-session-recall index', [
      'index',
      '--no-optimize',
    ]),
  );
  await system.writeFile(timerPath, renderSystemdTimer(interval));
  await system.writeFile(
    optimizeServicePath,
    renderSystemdService(system, 'Optimize the pi-session-recall index', ['optimize']),
  );
  await system.writeFile(optimizeTimerPath, renderSystemdOptimizeTimer());
  await runRequiredSchedulerProcess(system, 'systemctl', ['--user', 'daemon-reload']);
  await runRequiredSchedulerProcess(system, 'systemctl', [
    '--user',
    'enable',
    SYSTEMD_TIMER_NAME,
    SYSTEMD_OPTIMIZE_TIMER_NAME,
  ]);
  await runRequiredSchedulerProcess(system, 'systemctl', [
    '--user',
    'restart',
    SYSTEMD_TIMER_NAME,
    SYSTEMD_OPTIMIZE_TIMER_NAME,
  ]);

  const immediateResult = await system.runProcess('systemctl', [
    '--user',
    'start',
    SYSTEMD_SERVICE_NAME,
  ]);
  if (immediateResult.exitCode === 0) {
    return {};
  }
  const detail = immediateResult.stderr.trim();
  return {
    immediateRunWarning:
      detail.length === 0
        ? 'the immediate psr index attempt failed'
        : `the immediate psr index attempt failed: ${detail}`,
  };
}

// Runtime-untested on macOS: launchctl plist acceptance, RunAtLoad/StartInterval execution,
// no-overlap behavior, direct Node invocation, logs, durable recall config, and embedding access.
async function installLaunchAgentAutoIndexSchedule(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): Promise<AutoIndexInstallationResult> {
  const launchAgentsDirectory = join(system.homeDirectory, 'Library', 'LaunchAgents');
  const plistPath = join(launchAgentsDirectory, LAUNCH_AGENT_FILE_NAME);
  const optimizePlistPath = join(launchAgentsDirectory, LAUNCH_AGENT_OPTIMIZE_FILE_NAME);
  const logsDirectory = join(system.homeDirectory, '.pi', 'agent', 'logs');
  await unloadLaunchAgent(system, plistPath);
  await unloadLaunchAgent(system, optimizePlistPath);
  await system.makeDirectory(launchAgentsDirectory);
  await system.makeDirectory(logsDirectory);
  await system.writeFile(plistPath, renderLaunchAgentIndexPlist(interval, system));
  await system.writeFile(optimizePlistPath, renderLaunchAgentOptimizePlist(system));
  await system.setFileMode(plistPath, 0o600);
  await system.setFileMode(optimizePlistPath, 0o600);
  await runRequiredSchedulerProcess(system, 'launchctl', ['load', plistPath]);
  await runRequiredSchedulerProcess(system, 'launchctl', ['load', optimizePlistPath]);
  return {};
}

async function uninstallLaunchAgentAutoIndexSchedule(
  system: AutoIndexSchedulerSystem,
): Promise<void> {
  const launchAgentsDirectory = join(system.homeDirectory, 'Library', 'LaunchAgents');
  const plistPath = join(launchAgentsDirectory, LAUNCH_AGENT_FILE_NAME);
  const optimizePlistPath = join(launchAgentsDirectory, LAUNCH_AGENT_OPTIMIZE_FILE_NAME);
  await unloadLaunchAgent(system, plistPath);
  await unloadLaunchAgent(system, optimizePlistPath);
  await system.removeFile(plistPath);
  await system.removeFile(optimizePlistPath);
}

async function uninstallSystemdAutoIndexSchedule(system: AutoIndexSchedulerSystem): Promise<void> {
  const userUnitDirectory = join(readSystemdConfigHome(system), 'systemd', 'user');
  for (const timerName of [SYSTEMD_TIMER_NAME, SYSTEMD_OPTIMIZE_TIMER_NAME]) {
    await runSystemdUninstallProcess(system, ['--user', 'disable', '--now', timerName], timerName);
  }
  for (const serviceName of [SYSTEMD_SERVICE_NAME, SYSTEMD_OPTIMIZE_SERVICE_NAME]) {
    await runSystemdUninstallProcess(system, ['--user', 'stop', serviceName], serviceName);
  }
  await system.removeFile(join(userUnitDirectory, SYSTEMD_SERVICE_NAME));
  await system.removeFile(join(userUnitDirectory, SYSTEMD_TIMER_NAME));
  await system.removeFile(join(userUnitDirectory, SYSTEMD_OPTIMIZE_SERVICE_NAME));
  await system.removeFile(join(userUnitDirectory, SYSTEMD_OPTIMIZE_TIMER_NAME));
  await runRequiredSchedulerProcess(system, 'systemctl', ['--user', 'daemon-reload']);
}

/** Installs or replaces the current user's native automatic index schedule. */
export async function installAutoIndexSchedule(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): Promise<AutoIndexInstallationResult> {
  if (!isAbsolute(system.nodeExecutablePath) || !isAbsolute(system.packageRoot)) {
    throw new Error('Auto-index scheduler requires absolute Node and package paths');
  }
  if (system.platform === 'linux') {
    return await installSystemdAutoIndexSchedule(interval, system);
  }
  if (system.platform === 'darwin') {
    return await installLaunchAgentAutoIndexSchedule(interval, system);
  }
  throw new Error(`Auto-index scheduler is not supported on platform ${system.platform}`);
}

/** Disables and removes the current user's native automatic index schedule. */
export async function uninstallAutoIndexSchedule(system: AutoIndexSchedulerSystem): Promise<void> {
  if (system.platform === 'linux') {
    await uninstallSystemdAutoIndexSchedule(system);
    return;
  }
  if (system.platform === 'darwin') {
    await uninstallLaunchAgentAutoIndexSchedule(system);
    return;
  }
  throw new Error(`Auto-index scheduler is not supported on platform ${system.platform}`);
}
