import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { realpath, stat } from 'node:fs';
import { isAbsolute, normalize, relative, sep } from 'node:path';

import { RecallProjectIdentitySource } from './enums.js';

/** Version of trusted-invocation exact project filtering before every retrieval-channel limit. */
export const PROJECT_SCOPE_POLICY_VERSION = 1;

/** Version of lineage-first, exact-origin, and canonical Git project identity resolution. */
export const PROJECT_IDENTITY_POLICY_VERSION = 4;

/** Version of project identity and identity-source scalars persisted on recall evidence. */
export const PROJECT_IDENTITY_METADATA_SCHEMA_VERSION = 3;

/** Version of exact-root and deterministic-descendant project lineage assignment. */
export const PROJECT_LINEAGE_POLICY_VERSION = 1;

const GIT_ORIGIN_IDENTITY_PREFIX = 'git-origin:';
const GIT_COMMON_DIRECTORY_IDENTITY_PREFIX = 'git-common-directory:';
const NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX = 'non-git-session-origin:';
const HOSTED_GIT_PROTOCOLS = new Set(['git:', 'http:', 'https:', 'ssh:']);
declare const PROJECT_IDENTITY_BRAND: unique symbol;
declare const REPOSITORY_IDENTITY_BRAND: unique symbol;
const RECALL_PROJECT_LINEAGES_BRAND = Symbol('RecallProjectLineages');

/** Stable scalar used to enforce one exact recall project boundary. */
export type ProjectIdentity = string & { readonly [PROJECT_IDENTITY_BRAND]: true };

/** Canonical Git-origin or shared-common-directory project identity. */
export type RepositoryIdentity = ProjectIdentity & {
  readonly [REPOSITORY_IDENTITY_BRAND]: true;
};

/** Returns whether a value is already a canonical Git repository identity. */
export function isCanonicalRepositoryIdentity(value: string): value is RepositoryIdentity {
  if (value.startsWith(GIT_ORIGIN_IDENTITY_PREFIX)) {
    const origin = /^git-origin:([a-z0-9.-]+(?::[0-9]+)?)\/(.+)$/u.exec(value);
    if (!origin) {
      return false;
    }
    const repositoryPath = normalizeGitRepositoryPath(origin[2] ?? '');
    return (
      repositoryPath !== null &&
      value === `${GIT_ORIGIN_IDENTITY_PREFIX}${origin[1]}/${repositoryPath}`
    );
  }
  if (value.startsWith(GIT_COMMON_DIRECTORY_IDENTITY_PREFIX)) {
    const commonDirectory = value.slice(GIT_COMMON_DIRECTORY_IDENTITY_PREFIX.length);
    return isAbsolute(commonDirectory) && normalize(commonDirectory) === commonDirectory;
  }
  return false;
}

/** Raw repository-identity keys and historical roots before lineage validation. */
export interface RecallProjectLineageInput {
  readonly [repositoryIdentity: string]: readonly string[];
}

/** Validated project lineages keyed only by canonical repository identities. */
export interface RecallProjectLineages extends ReadonlyMap<RepositoryIdentity, readonly string[]> {
  readonly [RECALL_PROJECT_LINEAGES_BRAND]: true;
}

interface ProjectLineageDeclaration {
  repositoryIdentity: RepositoryIdentity;
  historicalRoot: string;
}

class ValidatedRecallProjectLineages
  extends Map<RepositoryIdentity, string[]>
  implements RecallProjectLineages
{
  readonly [RECALL_PROJECT_LINEAGES_BRAND]: true = true;
}

function isPathAtOrBelow(path: string, root: string): boolean {
  const relativePath = relative(root, path);
  return (
    relativePath === '' ||
    (relativePath !== '..' && !relativePath.startsWith(`..${sep}`) && !isAbsolute(relativePath))
  );
}

/** Validates, normalizes, and orders personal project lineage declarations deterministically. */
export function normalizeRecallProjectLineages(
  projectLineages: RecallProjectLineageInput,
): RecallProjectLineages {
  const declarations: ProjectLineageDeclaration[] = [];
  for (const [repositoryIdentity, configuredRoots] of Object.entries(projectLineages).toSorted(
    (left, right) => left[0].localeCompare(right[0]),
  )) {
    if (!isCanonicalRepositoryIdentity(repositoryIdentity)) {
      throw new Error(
        `Recall configuration project lineage target must be a canonical repository identity: ${repositoryIdentity}`,
      );
    }
    for (const configuredRoot of configuredRoots) {
      if (!isAbsolute(configuredRoot)) {
        throw new Error(
          `Recall configuration project lineage root must be absolute: ${configuredRoot}`,
        );
      }
      declarations.push({ repositoryIdentity, historicalRoot: normalize(configuredRoot) });
    }
  }
  declarations.sort(
    (left, right) =>
      left.historicalRoot.localeCompare(right.historicalRoot) ||
      left.repositoryIdentity.localeCompare(right.repositoryIdentity),
  );
  for (const [index, declaration] of declarations.entries()) {
    for (const other of declarations.slice(index + 1)) {
      if (
        declaration.repositoryIdentity !== other.repositoryIdentity &&
        (isPathAtOrBelow(declaration.historicalRoot, other.historicalRoot) ||
          isPathAtOrBelow(other.historicalRoot, declaration.historicalRoot))
      ) {
        throw new Error(
          `Recall configuration project lineage roots conflict: ${declaration.historicalRoot} maps to ${declaration.repositoryIdentity}, while ${other.historicalRoot} maps to ${other.repositoryIdentity}; assign them to one repository identity or remove the overlap`,
        );
      }
    }
  }
  const normalized = new ValidatedRecallProjectLineages();
  for (const declaration of declarations) {
    const roots = normalized.get(declaration.repositoryIdentity) ?? [];
    if (!roots.includes(declaration.historicalRoot)) {
      roots.push(declaration.historicalRoot);
    }
    normalized.set(declaration.repositoryIdentity, roots);
  }
  return normalized;
}

/** Hashes validated personal lineage declarations without consulting the filesystem. */
export function createLineageDigest(projectLineages: RecallProjectLineages): string {
  const serializableLineages = Object.fromEntries(projectLineages.entries());
  return createHash('sha256').update(JSON.stringify(serializableLineages)).digest('hex');
}

/** One exact project identity and the explicit source that established it. */
export type ResolvedProjectIdentity =
  | {
      projectIdentity: RepositoryIdentity;
      identitySource:
        | RecallProjectIdentitySource.GIT_ORIGIN
        | RecallProjectIdentitySource.GIT_COMMON_DIRECTORY
        | RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE;
    }
  | {
      projectIdentity: ProjectIdentity;
      identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN;
    };

/** Validates and brands a canonical repository identity from an external scalar boundary. */
export function parseRepositoryIdentity(value: string): RepositoryIdentity {
  if (!isCanonicalRepositoryIdentity(value)) {
    throw new Error(`Recall repository identity invalid: ${value}`);
  }
  return value;
}

/** Validates and brands a project identity read from an external scalar boundary. */
export function parseProjectIdentity(value: string): ProjectIdentity;
export function parseProjectIdentity(value: string): string {
  if (isCanonicalRepositoryIdentity(value)) {
    return value;
  }
  if (value.startsWith(NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX)) {
    const sessionOrigin = value.slice(NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX.length);
    if (isAbsolute(sessionOrigin) && normalize(sessionOrigin) === sessionOrigin) {
      return value;
    }
  }
  throw new Error(`Recall project identity invalid: ${value}`);
}

/** Resolves explicit project lineage before falling back to repository or exact-origin identity. */
export function createLineageResolver(
  projectLineages: RecallProjectLineages,
  fallbackResolver: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>,
): (workingDirectory: string) => Promise<ResolvedProjectIdentity | null> {
  const declarations = Array.from(projectLineages.entries())
    .flatMap(([repositoryIdentity, historicalRoots]) =>
      historicalRoots.map((historicalRoot) => ({ repositoryIdentity, historicalRoot })),
    )
    .sort(
      (left, right) =>
        right.historicalRoot.length - left.historicalRoot.length ||
        left.historicalRoot.localeCompare(right.historicalRoot),
    );
  return async (workingDirectory) => {
    if (isAbsolute(workingDirectory)) {
      const normalizedWorkingDirectory = normalize(workingDirectory);
      const lineage = declarations.find((declaration) =>
        isPathAtOrBelow(normalizedWorkingDirectory, declaration.historicalRoot),
      );
      if (lineage) {
        return {
          projectIdentity: lineage.repositoryIdentity,
          identitySource: RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE,
        };
      }
    }
    return fallbackResolver(workingDirectory);
  };
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

function normalizeHostedGitOrigin(remote: string): RepositoryIdentity | null {
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
    const identity =
      host && repositoryPath ? `${GIT_ORIGIN_IDENTITY_PREFIX}${host}/${repositoryPath}` : null;
    return identity && isCanonicalRepositoryIdentity(identity) ? identity : null;
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
  const identity = repositoryPath
    ? `${GIT_ORIGIN_IDENTITY_PREFIX}${parsedRemote.host.toLowerCase()}/${repositoryPath}`
    : null;
  return identity && isCanonicalRepositoryIdentity(identity) ? identity : null;
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
    projectIdentity: parseProjectIdentity(
      `${NON_GIT_SESSION_ORIGIN_IDENTITY_PREFIX}${workingDirectory}`,
    ),
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
  const commonDirectoryIdentity = realCommonDirectory
    ? `${GIT_COMMON_DIRECTORY_IDENTITY_PREFIX}${realCommonDirectory}`
    : null;
  return commonDirectoryIdentity && isCanonicalRepositoryIdentity(commonDirectoryIdentity)
    ? {
        projectIdentity: commonDirectoryIdentity,
        identitySource: RecallProjectIdentitySource.GIT_COMMON_DIRECTORY,
      }
    : null;
}
