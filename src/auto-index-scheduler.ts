import { spawn } from 'node:child_process';
import { chmod, mkdir, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SYSTEMD_SERVICE_NAME = 'pi-session-recall-index.service';
const SYSTEMD_TIMER_NAME = 'pi-session-recall-index.timer';
const LAUNCH_AGENT_LABEL = 'dev.pi-session-recall.auto-index';
const LAUNCH_AGENT_FILE_NAME = `${LAUNCH_AGENT_LABEL}.plist`;

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

function renderLaunchAgentPlist(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): string {
  const intervalSeconds = interval.value * (interval.unit === 'm' ? 60n : 3_600n);
  const logsDirectory = join(system.homeDirectory, '.pi', 'agent', 'logs');
  const argumentsList = [
    system.nodeExecutablePath,
    '--import',
    'tsx',
    join(system.packageRoot, 'bin', 'psr'),
    'index',
  ];
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    '  <key>Label</key>',
    `  <string>${LAUNCH_AGENT_LABEL}</string>`,
    '  <key>ProgramArguments</key>',
    '  <array>',
    ...argumentsList.map((argument) => `    <string>${escapePlistXml(argument)}</string>`),
    '  </array>',
    '  <key>WorkingDirectory</key>',
    `  <string>${escapePlistXml(system.packageRoot)}</string>`,
    '  <key>StartInterval</key>',
    `  <integer>${intervalSeconds}</integer>`,
    '  <key>RunAtLoad</key>',
    '  <true/>',
    '  <key>StandardOutPath</key>',
    `  <string>${escapePlistXml(join(logsDirectory, 'pi-session-recall-auto-index.out.log'))}</string>`,
    '  <key>StandardErrorPath</key>',
    `  <string>${escapePlistXml(join(logsDirectory, 'pi-session-recall-auto-index.err.log'))}</string>`,
    '</dict>',
    '</plist>',
    '',
  ].join('\n');
}

function renderSystemdService(system: AutoIndexSchedulerSystem): string {
  const executablePath = quoteSystemdValue(system.nodeExecutablePath);
  const packageRoot = quoteSystemdValue(system.packageRoot);
  const psrExecutablePath = quoteSystemdValue(join(system.packageRoot, 'bin', 'psr'));
  return [
    '[Unit]',
    'Description=Maintain the pi-session-recall index',
    '',
    '[Service]',
    'Type=oneshot',
    `WorkingDirectory=${packageRoot}`,
    `ExecStart=${executablePath} --import tsx ${psrExecutablePath} index`,
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

async function runRequiredSchedulerProcess(
  system: AutoIndexSchedulerSystem,
  executable: string,
  argumentsList: readonly string[],
): Promise<void> {
  const result = await system.runProcess(executable, argumentsList);
  if (result.exitCode !== 0) {
    const detail = result.stderr.trim();
    const suffix = detail.length === 0 ? '' : `: ${detail}`;
    throw new Error(
      `Auto-index scheduler command failed: ${executable} ${argumentsList.join(' ')}${suffix}`,
    );
  }
}

async function installSystemdAutoIndexSchedule(
  interval: AutoIndexInterval,
  system: AutoIndexSchedulerSystem,
): Promise<AutoIndexInstallationResult> {
  const userUnitDirectory = join(readSystemdConfigHome(system), 'systemd', 'user');
  const servicePath = join(userUnitDirectory, SYSTEMD_SERVICE_NAME);
  const timerPath = join(userUnitDirectory, SYSTEMD_TIMER_NAME);
  await system.makeDirectory(userUnitDirectory);
  await system.writeFile(servicePath, renderSystemdService(system));
  await system.writeFile(timerPath, renderSystemdTimer(interval));
  await runRequiredSchedulerProcess(system, 'systemctl', ['--user', 'daemon-reload']);
  await runRequiredSchedulerProcess(system, 'systemctl', ['--user', 'enable', SYSTEMD_TIMER_NAME]);
  await runRequiredSchedulerProcess(system, 'systemctl', ['--user', 'restart', SYSTEMD_TIMER_NAME]);

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
  const logsDirectory = join(system.homeDirectory, '.pi', 'agent', 'logs');
  await system.runProcess('launchctl', ['unload', plistPath]);
  await system.makeDirectory(launchAgentsDirectory);
  await system.makeDirectory(logsDirectory);
  await system.writeFile(plistPath, renderLaunchAgentPlist(interval, system));
  await system.setFileMode(plistPath, 0o600);
  await runRequiredSchedulerProcess(system, 'launchctl', ['load', plistPath]);
  return {};
}

async function uninstallLaunchAgentAutoIndexSchedule(
  system: AutoIndexSchedulerSystem,
): Promise<void> {
  const plistPath = join(system.homeDirectory, 'Library', 'LaunchAgents', LAUNCH_AGENT_FILE_NAME);
  await system.runProcess('launchctl', ['unload', plistPath]);
  await system.removeFile(plistPath);
}

async function uninstallSystemdAutoIndexSchedule(system: AutoIndexSchedulerSystem): Promise<void> {
  const userUnitDirectory = join(readSystemdConfigHome(system), 'systemd', 'user');
  await system.runProcess('systemctl', ['--user', 'disable', '--now', SYSTEMD_TIMER_NAME]);
  await system.runProcess('systemctl', ['--user', 'stop', SYSTEMD_SERVICE_NAME]);
  await system.removeFile(join(userUnitDirectory, SYSTEMD_SERVICE_NAME));
  await system.removeFile(join(userUnitDirectory, SYSTEMD_TIMER_NAME));
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
