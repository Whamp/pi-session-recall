import { execFile } from 'node:child_process';
import { realpath, stat } from 'node:fs';

import { RecallProjectIdentitySource } from './enums.js';

/** Version of exact non-Git origin and canonical Git repository identity resolution. */
export const RECALL_PROJECT_IDENTITY_POLICY_VERSION = 2;

/** Version of project identity and identity-source scalars persisted on recall evidence. */
export const RECALL_PROJECT_IDENTITY_METADATA_SCHEMA_VERSION = 2;

const GIT_ORIGIN_IDENTITY_PREFIX = 'git-origin:';
const GIT_COMMON_DIRECTORY_IDENTITY_PREFIX = 'git-common-directory:';
const NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX = 'non-git-session-origin:';
const HOSTED_GIT_PROTOCOLS = new Set(['git:', 'http:', 'https:', 'ssh:']);

/** One exact project identity and the explicit source that established it. */
export interface ResolvedProjectIdentity {
  projectIdentity: string;
  identitySource: RecallProjectIdentitySource;
}

function normalizeGitRepositoryPath(rawPath: string): string | null {
  const withoutOuterSlashes = rawPath.replace(/^\/+|\/+$/gu, '');
  const withoutGitSuffix = withoutOuterSlashes.replace(/\.git$/iu, '');
  const segments = withoutGitSuffix.split('/');
  if (
    segments.length < 2 ||
    segments.some((segment) => segment.length === 0 || segment === '.' || segment === '..')
  ) {
    return null;
  }
  return segments.join('/');
}

function normalizeHostedGitOrigin(remote: string): string | null {
  const trimmedRemote = remote.trim();
  if (!trimmedRemote) {
    return null;
  }
  const scpRemote = trimmedRemote.includes('://')
    ? null
    : /^(?:[^@/:\s]+@)?([^/:\s]+):(.+)$/u.exec(trimmedRemote);
  if (scpRemote) {
    const host = scpRemote[1]?.toLowerCase();
    const repositoryPath = normalizeGitRepositoryPath(scpRemote[2] ?? '');
    return host && repositoryPath ? `${GIT_ORIGIN_IDENTITY_PREFIX}${host}/${repositoryPath}` : null;
  }
  const parsedRemote = URL.parse(trimmedRemote);
  if (
    !parsedRemote ||
    !HOSTED_GIT_PROTOCOLS.has(parsedRemote.protocol) ||
    !parsedRemote.hostname ||
    parsedRemote.search ||
    parsedRemote.hash
  ) {
    return null;
  }
  const repositoryPath = normalizeGitRepositoryPath(parsedRemote.pathname);
  return repositoryPath
    ? `${GIT_ORIGIN_IDENTITY_PREFIX}${parsedRemote.hostname.toLowerCase()}/${repositoryPath}`
    : null;
}

function readGitOutput(workingDirectory: string, argumentsList: string[]): Promise<string | null> {
  return new Promise((resolve) => {
    execFile(
      'git',
      ['-C', workingDirectory, ...argumentsList],
      { encoding: 'utf8' },
      (error, stdout) => {
        if (error) {
          resolve(null);
          return;
        }
        const output = stdout.trim();
        resolve(output || null);
      },
    );
  });
}

function readRealGitCommonDirectory(commonDirectory: string): Promise<string | null> {
  return new Promise((resolve) => {
    realpath(commonDirectory, (error, resolvedPath) => {
      resolve(error ? null : resolvedPath);
    });
  });
}

function isExistingDirectory(workingDirectory: string): Promise<boolean> {
  return new Promise((resolve) => {
    stat(workingDirectory, (error, workingDirectoryStats) => {
      resolve(!error && workingDirectoryStats.isDirectory());
    });
  });
}

async function resolveExistingNonGitSessionOrigin(
  workingDirectory: string,
): Promise<ResolvedProjectIdentity | null> {
  if (!(await isExistingDirectory(workingDirectory))) {
    return null;
  }
  return {
    projectIdentity: `${NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX}${workingDirectory}`,
    identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
  };
}

/** Resolves Git repositories canonically and existing non-Git directories by exact session origin. */
export async function resolveProjectIdentity(
  workingDirectory: string,
): Promise<ResolvedProjectIdentity | null> {
  const origin = await readGitOutput(workingDirectory, ['config', '--get', 'remote.origin.url']);
  const canonicalOrigin = origin ? normalizeHostedGitOrigin(origin) : null;
  if (canonicalOrigin) {
    return {
      projectIdentity: canonicalOrigin,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    };
  }
  const commonDirectory = await readGitOutput(workingDirectory, [
    'rev-parse',
    '--path-format=absolute',
    '--git-common-dir',
  ]);
  if (!commonDirectory) {
    return resolveExistingNonGitSessionOrigin(workingDirectory);
  }
  const realCommonDirectory = await readRealGitCommonDirectory(commonDirectory);
  return realCommonDirectory
    ? {
        projectIdentity: `${GIT_COMMON_DIRECTORY_IDENTITY_PREFIX}${realCommonDirectory}`,
        identitySource: RecallProjectIdentitySource.GIT_COMMON_DIRECTORY,
      }
    : null;
}
