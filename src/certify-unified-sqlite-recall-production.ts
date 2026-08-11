import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  opendirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  statfsSync,
  statSync,
} from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';
import { gunzipSync } from 'node:zlib';

import { RecallSearchScope } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  indexChangedConversationSessions,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import { createOctenHttpEmbeddingProvider } from './octen-http-embedding-provider.js';
import { listIgnoredPhysicalSessionPaths } from './physical-session-ignore.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { readRecallIndexManifest } from './recall-index-manifest.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  createLineageResolver,
  parseProjectIdentity,
  resolveProjectIdentity,
} from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_VEC_PACKAGE_VERSION,
  type SqliteRecallDatabase,
  type SqliteRecallDatabaseCounts,
  type SqliteRecallIntegrityDiagnostics,
} from './sqlite-recall-database.js';
const GIBIBYTE = 1_024 ** 3;
const MEBIBYTE = 1_024 ** 2;
/** Maximum allocated bytes accepted for one complete production Recall database. */
export const MAXIMUM_CERTIFIED_STORAGE_BYTES = 5 * GIBIBYTE;
/** Maximum allocation permitted for one disposable certification clone. */
export const MAXIMUM_SCRATCH_ALLOCATION_BYTES = 6 * GIBIBYTE;
/** Minimum free bytes required before creating a disposable certification clone. */
export const MINIMUM_SCRATCH_FREE_BYTES = 240 * GIBIBYTE;
/** Maximum accepted warm project-scoped dense-search p95 in milliseconds. */
export const MAXIMUM_PROJECT_P95_MILLISECONDS = 100;
/** Maximum accepted warm global dense-search p95 in milliseconds. */
export const MAXIMUM_GLOBAL_P95_MILLISECONDS = 500;
/** Maximum accepted warm Invocation-search p95 in milliseconds. */
export const MAXIMUM_INVOCATION_P95_MILLISECONDS = 5;
/** Maximum device bytes written by one representative changed-session replacement. */
export const MAXIMUM_AVERAGE_DEVICE_WRITES_BYTES = 10 * MEBIBYTE;
const BENCHMARK_REPETITIONS = 6;
const REPORT_JSON_PATH = resolve(
  'docs/research/unified-sqlite-production-recall-certification.json',
);
const REPORT_MARKDOWN_PATH = resolve(
  'docs/research/unified-sqlite-production-recall-certification.md',
);
const USAGE =
  'Unified SQLite recall certification usage: --data-root <exact-path> --candidate-target generations/generation-... --control-zvec <exact-v7-flat-zvec-path> --project-identity <identity> [--scratch-root <exact-path> --representative-session <indexed-path> --block-device <name>] [--output docs/research/unified-sqlite-production-recall-certification.json]';

/** Fixed production-derived questions used for dense latency and overlap certification. */
export const DENSE_CERTIFICATION_QUERIES = [
  'Why have recent pi-session-recall optimization attempts failed?',
  'How is automatic recall indexing scheduled?',
  'Which corrupted February session files are ignored?',
  'How large is the recall database?',
  'Why would an agent use pi-session-recall instead of searching raw JSONL?',
] as const;

/** Fixed locator classes that every production Invocation FTS projection must retrieve. */
export const INVOCATION_CERTIFICATION_PROBES = [
  { kind: 'tool-name', query: 'brain_query', expectedToolName: 'brain_query' },
  { kind: 'path', query: '/home/will/.pi/agent/TAILNET.md' },
  { kind: 'url', query: 'http://192.168.0.67:8090/v1' },
  { kind: 'command', query: 'psr optimize' },
  { kind: 'issue-number', query: 'gh issue view 165' },
  { kind: 'flag', query: '--optimize-daily' },
] as const;

/** Fixed raw-payload evidence that explicit Source search must locate with provenance. */
export const SOURCE_CERTIFICATION_PROBES = [
  { kind: 'result-only-error', query: 'FtsRocksdbReducer', requiredRole: 'toolResult' },
  { kind: 'hardware-identifier', query: 'CT1000P3PSSD8', requiredRole: 'toolResult' },
  { kind: 'filename', query: '2026-02-02T18-31-25' },
  {
    kind: 'command-output',
    query: 'pi - AI coding assistant with read, bash, edit, write tools',
    requiredRole: 'bashExecution',
  },
] as const;

/** Exact operator inputs for read-only candidate checks and optional clone-only mutation checks. */
export interface UnifiedSqliteCertificationArguments {
  dataRoot: string;
  candidateTarget: string;
  controlZvecPath: string;
  projectIdentity: string;
  scratchRoot: string | null;
  representativeSessionPath: string | null;
  blockDevice: string | null;
  outputPath: string | null;
}

/** Gate inputs kept separate so ordinary tests can prove the production thresholds. */
export interface UnifiedSqliteCertificationGateInputs {
  storageBytes: number;
  projectP95Milliseconds: number;
  globalP95Milliseconds: number;
  invocationP95Milliseconds: number;
  denseTopResultsMatch: readonly boolean[];
  globalTopEightOverlaps: readonly number[];
  invocationProbePasses: readonly boolean[];
  sourceProbePasses: readonly boolean[];
  integrityHealthy: boolean;
  linuxX64LoadPassed: boolean;
  macOsPackagesAvailable: boolean;
  macOsX64RuntimeLoadPassed: boolean | null;
  macOsArm64RuntimeLoadPassed: boolean | null;
  candidateInactive: boolean;
  clonePassed: boolean | null;
}

/** Evaluates every pre-activation threshold without reading production state. */
export function evaluateUnifiedSqliteCertificationGates(
  inputs: UnifiedSqliteCertificationGateInputs,
): Record<string, boolean | null> {
  return {
    storage: inputs.storageBytes <= MAXIMUM_CERTIFIED_STORAGE_BYTES,
    projectLatency: inputs.projectP95Milliseconds < MAXIMUM_PROJECT_P95_MILLISECONDS,
    globalLatency: inputs.globalP95Milliseconds < MAXIMUM_GLOBAL_P95_MILLISECONDS,
    invocationLatency: inputs.invocationP95Milliseconds < MAXIMUM_INVOCATION_P95_MILLISECONDS,
    invocationProbes:
      inputs.invocationProbePasses.length === INVOCATION_CERTIFICATION_PROBES.length &&
      inputs.invocationProbePasses.every(Boolean),
    denseTopResults:
      inputs.denseTopResultsMatch.length === DENSE_CERTIFICATION_QUERIES.length &&
      inputs.denseTopResultsMatch.every(Boolean),
    globalOverlap:
      inputs.globalTopEightOverlaps.length === DENSE_CERTIFICATION_QUERIES.length &&
      inputs.globalTopEightOverlaps.every((overlap) => overlap >= 7),
    sourceProvenance:
      inputs.sourceProbePasses.length === SOURCE_CERTIFICATION_PROBES.length &&
      inputs.sourceProbePasses.every(Boolean),
    integrity: inputs.integrityHealthy,
    linuxX64Load: inputs.linuxX64LoadPassed,
    macOsPackageAvailability: inputs.macOsPackagesAvailable,
    macOsX64RuntimeLoad: inputs.macOsX64RuntimeLoadPassed,
    macOsArm64RuntimeLoad: inputs.macOsArm64RuntimeLoadPassed,
    candidateInactive: inputs.candidateInactive,
    clone: inputs.clonePassed,
  };
}

/** Parses exact paths before any production or candidate state is opened. */
export function readUnifiedSqliteCertificationArguments(
  argumentsList: readonly string[],
): UnifiedSqliteCertificationArguments {
  const values = new Map<string, string>();
  for (let index = 0; index < argumentsList.length; index += 2) {
    const flag = argumentsList[index];
    const value = argumentsList[index + 1];
    if (!flag || !value || !flag.startsWith('--') || values.has(flag)) {
      throw new Error(USAGE);
    }
    values.set(flag, value);
  }
  const allowed = new Set([
    '--data-root',
    '--candidate-target',
    '--control-zvec',
    '--project-identity',
    '--scratch-root',
    '--representative-session',
    '--block-device',
    '--output',
  ]);
  if ([...values.keys()].some((flag) => !allowed.has(flag))) {
    throw new Error(USAGE);
  }
  const required = ['--data-root', '--candidate-target', '--control-zvec', '--project-identity'];
  if (required.some((flag) => !values.get(flag))) {
    throw new Error(USAGE);
  }
  const scratchFlags = ['--scratch-root', '--representative-session', '--block-device'];
  const suppliedScratchFlags = scratchFlags.filter((flag) => values.has(flag));
  if (suppliedScratchFlags.length !== 0 && suppliedScratchFlags.length !== scratchFlags.length) {
    throw new Error(
      'Unified SQLite recall clone certification requires --scratch-root, --representative-session, and --block-device together',
    );
  }
  for (const pathFlag of [
    '--data-root',
    '--control-zvec',
    '--scratch-root',
    '--representative-session',
  ]) {
    const pathValue = values.get(pathFlag);
    if (pathValue && !isAbsolute(pathValue)) {
      throw new Error(
        `Unified SQLite recall certification requires an absolute path for ${pathFlag}`,
      );
    }
  }
  const blockDevice = values.get('--block-device') ?? null;
  if (blockDevice && !/^[A-Za-z0-9._-]+$/u.test(blockDevice)) {
    throw new Error(
      `Unified SQLite recall certification block device name is invalid: ${blockDevice}`,
    );
  }
  const outputPath = values.get('--output') ? resolve(values.get('--output') ?? '') : null;
  if (outputPath && outputPath !== REPORT_JSON_PATH) {
    throw new Error(
      'Unified SQLite recall certification output must be docs/research/unified-sqlite-production-recall-certification.json',
    );
  }
  return {
    dataRoot: resolve(values.get('--data-root') ?? ''),
    candidateTarget: values.get('--candidate-target') ?? '',
    controlZvecPath: resolve(values.get('--control-zvec') ?? ''),
    projectIdentity: values.get('--project-identity') ?? '',
    scratchRoot: values.get('--scratch-root') ? resolve(values.get('--scratch-root') ?? '') : null,
    representativeSessionPath: values.get('--representative-session')
      ? resolve(values.get('--representative-session') ?? '')
      : null,
    blockDevice,
    outputPath,
  };
}

/** Resolves only one exact generation target and rejects aliases, traversal, and active storage. */
export function resolveCertifiedCandidateDirectory(
  dataRoot: string,
  candidateTarget: string,
): string {
  const normalizedTarget = candidateTarget.replaceAll('\\', '/');
  if (
    normalizedTarget !== candidateTarget ||
    !/^generations\/generation-[A-Za-z0-9][A-Za-z0-9-]*$/u.test(candidateTarget)
  ) {
    throw new Error(
      `Unified SQLite recall candidate target is not an exact staged generation: ${candidateTarget}`,
    );
  }
  const candidateDirectory = resolve(dataRoot, candidateTarget);
  if (relative(dataRoot, candidateDirectory) !== candidateTarget) {
    throw new Error(
      `Unified SQLite recall candidate target escapes the data root: ${candidateTarget}`,
    );
  }
  return candidateDirectory;
}

/** Rejects scratch roots that could overlap the candidate, active data root, or filesystem root. */
export function assertCertificationScratchRoot(
  scratchRoot: string,
  dataRoot: string,
  candidateDirectory: string,
): void {
  const scratch = resolve(scratchRoot);
  const data = resolve(dataRoot);
  const candidate = resolve(candidateDirectory);
  const contains = (parent: string, child: string): boolean => {
    const path = relative(parent, child);
    return path === '' || (!path.startsWith(`..${sep}`) && path !== '..');
  };
  if (
    scratch === resolve('/') ||
    contains(scratch, data) ||
    contains(data, scratch) ||
    contains(scratch, candidate) ||
    contains(candidate, scratch)
  ) {
    throw new Error(
      `Unified SQLite recall scratch root must be disposable and disjoint: ${scratch}`,
    );
  }
}

function readAllocatedBytes(path: string): number {
  if (!existsSync(path)) {
    return 0;
  }
  const stats = statSync(path);
  if (stats.isFile()) {
    return stats.blocks * 512;
  }
  let bytes = 0;
  const directory = opendirSync(path);
  try {
    for (;;) {
      const entry = directory.readSync();
      if (!entry) {
        break;
      }
      bytes += readAllocatedBytes(join(path, entry.name));
    }
  } finally {
    directory.closeSync();
  }
  return bytes;
}

function readDatabaseAllocatedBytes(databasePath: string): number {
  return ['', '-wal', '-shm'].reduce(
    (bytes, suffix) => bytes + readAllocatedBytes(`${databasePath}${suffix}`),
    0,
  );
}

function readPercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    throw new Error('Unified SQLite recall latency sample is empty');
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(percentile * sorted.length) - 1)] ?? 0;
}

interface ActivePointerSnapshot {
  exists: boolean;
  target: string | null;
  bytesSha256: string | null;
}

/** Captures active pointer bytes and target without following it. */
export function snapshotUnifiedSqliteActivePointer(dataRoot: string): ActivePointerSnapshot {
  const pointerPath = join(dataRoot, 'active');
  let stats: ReturnType<typeof lstatSync>;
  try {
    stats = lstatSync(pointerPath);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return { exists: false, target: null, bytesSha256: null };
    }
    throw error;
  }
  const target = stats.isSymbolicLink() ? readlinkSync(pointerPath) : null;
  const bytes = Buffer.from(target ?? readFileSync(pointerPath));
  return {
    exists: true,
    target,
    bytesSha256: createHash('sha256').update(bytes).digest('hex'),
  };
}

function readSourceRole(sessionPath: string, sourceLine: number): string | null {
  const line = readFileSync(sessionPath, 'utf8').split('\n')[sourceLine - 1];
  if (!line) {
    return null;
  }
  const value: unknown = JSON.parse(line);
  if (!isUnknownRecord(value)) {
    return null;
  }
  if (value.type === 'message' && isUnknownRecord(value.message)) {
    return typeof value.message.role === 'string' ? value.message.role : null;
  }
  return typeof value.type === 'string' ? value.type : null;
}

function readTarHeaderString(value: Uint8Array): string {
  const text = Buffer.from(value).toString('utf8');
  const terminator = text.indexOf('\0');
  return terminator < 0 ? text : text.slice(0, terminator);
}

function inspectTarballFileNames(tarGzip: Uint8Array): string[] {
  const tar = gunzipSync(tarGzip);
  const names: string[] = [];
  for (let offset = 0; offset + 512 <= tar.length; ) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) {
      break;
    }
    const name = readTarHeaderString(header.subarray(0, 100));
    const sizeText = readTarHeaderString(header.subarray(124, 136)).trim();
    const size = Number.parseInt(sizeText || '0', 8);
    names.push(name);
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return names;
}

async function inspectPublishedSqliteVecPackage(
  packageName: string,
): Promise<Record<string, unknown>> {
  const metadataText = execFileSync(
    'npm',
    [
      'view',
      `${packageName}@${SQLITE_RECALL_VEC_PACKAGE_VERSION}`,
      'version',
      'dist.tarball',
      '--json',
    ],
    { encoding: 'utf8' },
  );
  const metadata: unknown = JSON.parse(metadataText);
  if (!isUnknownRecord(metadata)) {
    throw new Error(`sqlite-vec package metadata invalid for ${packageName}`);
  }
  const tarballUrl = metadata['dist.tarball'];
  if (typeof tarballUrl !== 'string' || !tarballUrl) {
    throw new Error(`sqlite-vec package metadata missing tarball for ${packageName}`);
  }
  const response = await fetch(tarballUrl);
  if (!response.ok) {
    throw new Error(
      `sqlite-vec package tarball fetch failed for ${packageName}: HTTP ${response.status}`,
    );
  }
  const files = inspectTarballFileNames(new Uint8Array(await response.arrayBuffer()));
  return {
    packageName,
    version: typeof metadata.version === 'string' ? metadata.version : null,
    tarballHost: new URL(tarballUrl).host,
    nativeFiles: files.filter((file) => /\.(dylib|so|dll)$/u.test(file)),
    packageAvailable:
      metadata.version === SQLITE_RECALL_VEC_PACKAGE_VERSION &&
      files.some((file) => file.endsWith('vec0.dylib')),
    inspectionPurpose: 'package-availability-only',
    executedLoad: false,
    executionStatus: 'not-executed; runtime load requires the matching macOS GitHub runner',
  };
}

function readBlockDeviceWrittenBytes(blockDevice: string): number | null {
  if (process.platform !== 'linux') {
    return null;
  }
  const path = `/sys/class/block/${blockDevice}/stat`;
  if (!existsSync(path)) {
    return null;
  }
  const fields = readFileSync(path, 'utf8').trim().split(/\s+/u);
  const sectors = Number(fields[6]);
  return Number.isFinite(sectors) ? sectors * 512 : null;
}

function runCloneChild(
  databasePath: string,
  sessionPath: string,
  mode: 'reader' | 'sigkill',
): { status: number | null; signal: NodeJS.Signals | null; stdout: string } {
  const child = spawnSync(
    process.execPath,
    ['--import', 'tsx', import.meta.filename, '--clone-child', mode, databasePath, sessionPath],
    { encoding: 'utf8' },
  );
  return { status: child.status, signal: child.signal, stdout: child.stdout };
}

function runCloneChildMode(argumentsList: readonly string[]): boolean {
  if (argumentsList[0] !== '--clone-child') {
    return false;
  }
  const mode = argumentsList[1];
  const databasePath = argumentsList[2];
  const sessionPath = argumentsList[3];
  if (!databasePath || !sessionPath || (mode !== 'reader' && mode !== 'sigkill')) {
    throw new Error(USAGE);
  }
  if (mode === 'reader') {
    const reader = openSqliteRecallDatabase(databasePath, { readOnly: true });
    try {
      process.stdout.write(JSON.stringify(reader.readPhysicalSessionState(sessionPath)));
    } finally {
      reader.close();
    }
    return true;
  }
  const writer = openSqliteRecallDatabase(databasePath);
  const replacement = writer.readPhysicalSessionReplacement(sessionPath);
  if (!replacement) {
    throw new Error(`Unified SQLite recall representative session is not indexed: ${sessionPath}`);
  }
  writer.replacePhysicalSession(replacement, {
    beforeCommit() {
      process.kill(process.pid, 'SIGKILL');
    },
  });
  throw new Error('Unified SQLite recall forced-termination child survived SIGKILL');
}

function snapshotPhysicalSessionProjection(
  replacement: ReturnType<SqliteRecallDatabase['readPhysicalSessionReplacement']>,
): string {
  if (!replacement) {
    return 'null';
  }
  return JSON.stringify({
    ...replacement,
    denseEmbeddings: [...replacement.denseEmbeddings.entries()].sort(([left], [right]) =>
      left.localeCompare(right),
    ),
  });
}

/** Runs every mutation probe against a disposable candidate clone and removes it afterward. */
export async function certifyDisposableUnifiedSqliteClone(options: {
  candidateDirectory: string;
  candidateDatabasePath: string;
  scratchRoot: string;
  dataRoot: string;
  representativeSessionPath: string;
  blockDevice: string;
  runChangedSessionIndex?: (
    database: SqliteRecallDatabase,
    onProgress: (event: RecallIndexProgressEvent) => void,
  ) => Promise<ConversationIndexSummary>;
  /** Test seam; production callers must use the 240 GiB default. */
  minimumFreeBytes?: number;
  /** Test seam; production callers must read the named block device. */
  readDeviceWrittenBytes?: () => number | null;
}): Promise<Record<string, unknown>> {
  if (typeof options.runChangedSessionIndex !== 'function') {
    throw new Error('Unified SQLite recall real changed-session index callback is required');
  }
  assertCertificationScratchRoot(options.scratchRoot, options.dataRoot, options.candidateDirectory);
  mkdirSync(options.scratchRoot, { recursive: true });
  const scratchStats = statfsSync(options.scratchRoot);
  const freeBytes = scratchStats.bavail * scratchStats.bsize;
  if (freeBytes < (options.minimumFreeBytes ?? MINIMUM_SCRATCH_FREE_BYTES)) {
    throw new Error('Unified SQLite recall scratch free space is below the 240 GiB floor');
  }
  const candidateBytes = readAllocatedBytes(options.candidateDirectory);
  if (candidateBytes > MAXIMUM_SCRATCH_ALLOCATION_BYTES) {
    throw new Error('Unified SQLite recall candidate exceeds the 6 GiB scratch allocation ceiling');
  }
  const cloneDirectory = join(options.scratchRoot, `certification-clone-${process.pid}`);
  let concurrentReaderState: unknown = null;
  let stateBeforeRollback: unknown = null;
  rmSync(cloneDirectory, { recursive: true, force: true });
  try {
    cpSync(options.candidateDirectory, cloneDirectory, { recursive: true, errorOnExist: true });
    if (readAllocatedBytes(cloneDirectory) > MAXIMUM_SCRATCH_ALLOCATION_BYTES) {
      throw new Error('Unified SQLite recall clone exceeded the 6 GiB scratch allocation ceiling');
    }
    const databasePath = join(cloneDirectory, basename(options.candidateDatabasePath));
    const database = openSqliteRecallDatabase(databasePath);
    const replacement = database.readPhysicalSessionReplacement(options.representativeSessionPath);
    if (!replacement) {
      database.close();
      throw new Error(
        `Unified SQLite recall representative session is not indexed: ${options.representativeSessionPath}`,
      );
    }
    stateBeforeRollback = database.readPhysicalSessionState(options.representativeSessionPath);
    try {
      database.replacePhysicalSession(replacement, {
        beforeCommit() {
          const reader = runCloneChild(databasePath, options.representativeSessionPath, 'reader');
          if (reader.status !== 0) {
            throw new Error('Unified SQLite recall concurrent reader probe failed');
          }
          concurrentReaderState = JSON.parse(reader.stdout);
          throw new Error('explicit-certification-rollback');
        },
      });
      throw new Error('Unified SQLite recall explicit rollback probe committed unexpectedly');
    } finally {
      database.close();
    }
  } catch (error) {
    if (!(error instanceof Error) || error.message !== 'explicit-certification-rollback') {
      rmSync(cloneDirectory, { recursive: true, force: true });
      throw error;
    }
  }

  try {
    const databasePath = join(cloneDirectory, basename(options.candidateDatabasePath));
    let database = openSqliteRecallDatabase(databasePath);
    const replacement = database.readPhysicalSessionReplacement(options.representativeSessionPath);
    if (!replacement) {
      throw new Error('Unified SQLite recall clone lost its representative session');
    }
    const unrelatedSessionPath = database
      .listPhysicalSessionPaths()
      .find((sessionPath) => sessionPath !== options.representativeSessionPath);
    if (!unrelatedSessionPath) {
      throw new Error('Unified SQLite recall clone requires an unrelated indexed physical session');
    }
    const beforeState = database.readPhysicalSessionState(options.representativeSessionPath);
    const afterRollbackState = database.readPhysicalSessionState(options.representativeSessionPath);
    database.close();
    const crash = runCloneChild(databasePath, options.representativeSessionPath, 'sigkill');
    database = openSqliteRecallDatabase(databasePath);
    const afterCrashState = database.readPhysicalSessionState(options.representativeSessionPath);

    // Stale only clone-owned state. Canonical JSONL, candidate storage, and active storage stay read-only.
    const canonicalSessionSize = statSync(options.representativeSessionPath).size;
    database.replacePhysicalSession({ ...replacement, size: canonicalSessionSize + 1 });
    const expectedVectorReuseCount = replacement.denseDocuments.length;
    const unrelatedProjectionBefore = snapshotPhysicalSessionProjection(
      database.readPhysicalSessionReplacement(unrelatedSessionPath),
    );
    const countsBeforeChangedSessionIndex = database.readCounts();
    database.checkpointDisposableClone();
    const readDeviceWrittenBytes =
      options.readDeviceWrittenBytes ?? (() => readBlockDeviceWrittenBytes(options.blockDevice));
    const changedSessionIndexWritesBefore = readDeviceWrittenBytes();
    const changedSessionIndexedPhysicalSessionPaths = new Set<string>();
    const changedSessionIndexStarted = performance.now();
    const changedSessionIndexSummary = await options.runChangedSessionIndex(database, (event) => {
      if (event.kind === 'indexing-maintenance-workset') {
        changedSessionIndexedPhysicalSessionPaths.add(event.sessionPath);
      }
    });
    const changedSessionIndexElapsedMilliseconds = performance.now() - changedSessionIndexStarted;
    database.checkpointDisposableClone();
    const changedSessionIndexWritesAfter = readDeviceWrittenBytes();
    const changedSessionIndexDeviceWrittenBytes =
      changedSessionIndexWritesBefore === null || changedSessionIndexWritesAfter === null
        ? null
        : changedSessionIndexWritesAfter - changedSessionIndexWritesBefore;
    const unrelatedProjectionAfter = snapshotPhysicalSessionProjection(
      database.readPhysicalSessionReplacement(unrelatedSessionPath),
    );
    const countsAfterChangedSessionIndex = database.readCounts();
    const unrelatedPhysicalSessionUnchanged =
      unrelatedProjectionBefore === unrelatedProjectionAfter;
    const databaseCountsUnchanged =
      JSON.stringify(countsBeforeChangedSessionIndex) ===
      JSON.stringify(countsAfterChangedSessionIndex);
    const indexedPhysicalSessionPaths = [...changedSessionIndexedPhysicalSessionPaths].sort();
    const changedSessionIndexSummaryValid =
      changedSessionIndexSummary.indexedSessions === 1 &&
      changedSessionIndexSummary.removedSessions === 0 &&
      changedSessionIndexSummary.failedSessions.length === 0 &&
      indexedPhysicalSessionPaths.length === 1 &&
      indexedPhysicalSessionPaths[0] === options.representativeSessionPath;
    const changedSessionIndexReusedExpectedVectors =
      expectedVectorReuseCount > 0 &&
      changedSessionIndexSummary.reusedVectors === expectedVectorReuseCount &&
      changedSessionIndexSummary.newlyEmbeddedChunks === 0 &&
      changedSessionIndexSummary.embeddingRequestCount === 0;

    const churnReplacement = database.readPhysicalSessionReplacement(
      options.representativeSessionPath,
    );
    if (!churnReplacement) {
      throw new Error(
        'Unified SQLite recall changed-session index removed its representative session',
      );
    }
    database.checkpointDisposableClone();
    const beforeMetrics = {
      allocatedBytes: readDatabaseAllocatedBytes(databasePath),
      storage: database.readStorageMetrics(),
    };
    const directDatabaseChurnWritesBefore = readDeviceWrittenBytes();
    const directDatabaseChurnStarted = performance.now();
    for (let cycle = 0; cycle < 100; cycle += 1) {
      database.replacePhysicalSession(churnReplacement);
    }
    const directDatabaseChurnElapsedMilliseconds = performance.now() - directDatabaseChurnStarted;
    database.checkpointDisposableClone();
    const directDatabaseChurnWritesAfter = readDeviceWrittenBytes();
    const afterMetrics = {
      allocatedBytes: readDatabaseAllocatedBytes(databasePath),
      storage: database.readStorageMetrics(),
    };
    const directDatabaseChurnDeviceWrittenBytes =
      directDatabaseChurnWritesBefore === null || directDatabaseChurnWritesAfter === null
        ? null
        : directDatabaseChurnWritesAfter - directDatabaseChurnWritesBefore;
    const integrity = database.checkIntegrity();
    const globalLatencySamples: number[] = [];
    const projectLatencySamples: number[] = [];
    const invocationLatencySamples: number[] = [];
    const embedding = [...churnReplacement.denseEmbeddings.values()][0];
    const projectIdentity = churnReplacement.denseDocuments[0]?.projectAttribution?.projectIdentity;
    if (embedding) {
      for (let index = 0; index < BENCHMARK_REPETITIONS; index += 1) {
        let sampleStarted = performance.now();
        database.searchDenseCandidates(embedding, 8);
        const globalElapsed = performance.now() - sampleStarted;
        sampleStarted = performance.now();
        database.searchDenseCandidates(embedding, 8, projectIdentity);
        const projectElapsed = performance.now() - sampleStarted;
        sampleStarted = performance.now();
        database.searchInvocations(INVOCATION_CERTIFICATION_PROBES[0].query, 20, projectIdentity);
        const invocationElapsed = performance.now() - sampleStarted;
        if (index > 0) {
          globalLatencySamples.push(globalElapsed);
          projectLatencySamples.push(projectElapsed);
          invocationLatencySamples.push(invocationElapsed);
        }
      }
    }
    database.close();
    const allocatedGrowthBytes = afterMetrics.allocatedBytes - beforeMetrics.allocatedBytes;
    const freePageGrowth = afterMetrics.storage.freePageCount - beforeMetrics.storage.freePageCount;
    const readerSawCommittedState =
      JSON.stringify(stateBeforeRollback) === JSON.stringify(concurrentReaderState);
    const passed =
      readerSawCommittedState &&
      JSON.stringify(beforeState) === JSON.stringify(afterRollbackState) &&
      JSON.stringify(beforeState) === JSON.stringify(afterCrashState) &&
      crash.signal === 'SIGKILL' &&
      changedSessionIndexSummaryValid &&
      changedSessionIndexReusedExpectedVectors &&
      unrelatedPhysicalSessionUnchanged &&
      databaseCountsUnchanged &&
      integrity.healthy &&
      integrity.invocationFtsIntegrityChecked &&
      embedding !== undefined &&
      projectIdentity !== undefined &&
      globalLatencySamples.length > 0 &&
      readPercentile(globalLatencySamples, 0.95) < MAXIMUM_GLOBAL_P95_MILLISECONDS &&
      projectLatencySamples.length > 0 &&
      readPercentile(projectLatencySamples, 0.95) < MAXIMUM_PROJECT_P95_MILLISECONDS &&
      invocationLatencySamples.length > 0 &&
      readPercentile(invocationLatencySamples, 0.95) < MAXIMUM_INVOCATION_P95_MILLISECONDS &&
      allocatedGrowthBytes === 0 &&
      freePageGrowth <= 0 &&
      changedSessionIndexDeviceWrittenBytes !== null &&
      changedSessionIndexDeviceWrittenBytes < MAXIMUM_AVERAGE_DEVICE_WRITES_BYTES &&
      directDatabaseChurnDeviceWrittenBytes !== null &&
      directDatabaseChurnDeviceWrittenBytes / 100 < MAXIMUM_AVERAGE_DEVICE_WRITES_BYTES;
    return {
      cloneDirectory,
      representativeSession: basename(options.representativeSessionPath),
      unrelatedPhysicalSession: basename(unrelatedSessionPath),
      concurrentReaderSawCommittedState: readerSawCommittedState,
      explicitRollbackRestoredState:
        JSON.stringify(beforeState) === JSON.stringify(afterRollbackState),
      forcedTerminationSignal: crash.signal,
      forcedTerminationRestoredState:
        JSON.stringify(beforeState) === JSON.stringify(afterCrashState),
      changedSessionIndexer: 'indexChangedConversationSessions',
      changedSessionIndexedPhysicalSessionPaths: indexedPhysicalSessionPaths,
      changedSessionIndexSummary,
      changedSessionIndexSummaryValid,
      changedSessionIndexExpectedVectorReuseCount: expectedVectorReuseCount,
      changedSessionIndexReusedExpectedVectors,
      changedSessionIndexElapsedMilliseconds,
      changedSessionIndexDeviceWrittenBytes,
      unrelatedPhysicalSessionUnchanged,
      databaseCountsUnchanged,
      countsBeforeChangedSessionIndex,
      countsAfterChangedSessionIndex,
      directDatabaseChurnProbe: true,
      directDatabaseChurnCycles: 100,
      directDatabaseChurnElapsedMilliseconds,
      directDatabaseChurnDeviceWrittenBytes,
      directDatabaseChurnAverageDeviceWrittenBytesPerCycle:
        directDatabaseChurnDeviceWrittenBytes === null
          ? null
          : directDatabaseChurnDeviceWrittenBytes / 100,
      deviceWriteMeasurement:
        changedSessionIndexDeviceWrittenBytes === null ||
        directDatabaseChurnDeviceWrittenBytes === null
          ? 'unavailable'
          : 'gross-block-device-writes',
      allocatedGrowthBytes,
      freePageGrowth,
      beforeDirectDatabaseChurnStorage: beforeMetrics,
      afterDirectDatabaseChurnStorage: afterMetrics,
      postChurnIntegrity: integrity,
      postChurnGlobalP95Milliseconds: globalLatencySamples.length
        ? readPercentile(globalLatencySamples, 0.95)
        : null,
      postChurnProjectP95Milliseconds: projectLatencySamples.length
        ? readPercentile(projectLatencySamples, 0.95)
        : null,
      postChurnInvocationP95Milliseconds: invocationLatencySamples.length
        ? readPercentile(invocationLatencySamples, 0.95)
        : null,
      passed,
    };
  } finally {
    rmSync(cloneDirectory, { recursive: true, force: true });
  }
}

/** Replaces machine-specific roots and strips excerpts before a report becomes durable. */
export function sanitizeUnifiedSqliteCertificationReport(
  value: unknown,
  replacements: Readonly<Record<string, string>> = {},
): unknown {
  const pathReplacements = {
    [homedir()]: '$HOME',
    ...replacements,
  };
  const sanitizeString = (input: string): string => {
    let output = input;
    for (const [raw, replacement] of Object.entries(pathReplacements).sort(
      ([left], [right]) => right.length - left.length,
    )) {
      if (raw) {
        output = output.replaceAll(raw, replacement);
      }
    }
    return output;
  };
  if (typeof value === 'string') {
    return sanitizeString(value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeUnifiedSqliteCertificationReport(item, pathReplacements));
  }
  if (isUnknownRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !['text', 'searchableText', 'embedding'].includes(key))
        .map(([key, item]) => [
          key,
          sanitizeUnifiedSqliteCertificationReport(item, pathReplacements),
        ]),
    );
  }
  return value;
}

function formatCertificationMarkdown(report: Record<string, unknown>): string {
  if (!isUnknownRecord(report.gates)) {
    throw new Error('Unified SQLite recall certification report gates are missing');
  }
  const rows = Object.entries(report.gates)
    .map(
      ([gate, passed]) => `| ${gate} | ${passed === null ? 'PENDING' : passed ? 'PASS' : 'FAIL'} |`,
    )
    .join('\n');
  const candidateVerdict = report.candidatePreActivationPassed
    ? 'The candidate passed every local pre-activation gate.'
    : 'One or more local pre-activation gates failed or remain pending.';
  return `# Unified SQLite production recall certification\n\nThis report records pre-activation checks only. The harness did not activate the candidate.\n\n## Verdict\n\n${candidateVerdict} Release readiness is **${report.releaseReady ? 'ready' : 'not ready'}**.\n\n## Gates\n\n| Gate | Result |\n| --- | --- |\n${rows}\n\nTarball inspection records package availability only; it does not prove runtime loading. The stable \`SQLite-vec macOS x64 runtime load\` and \`SQLite-vec macOS arm64 runtime load\` jobs in \`.github/workflows/sqlite-vec-platform-smoke.yml\` supply that external evidence when a PR workflow run succeeds. Record the real run URL after it exists; this report does not fabricate one.\n\n## Activation status\n\nActivation, immediate post-activation indexing, live project/global/Source recall, timer verification, and rollback remain pending until this branch is merged and deployed. macOS x64 and arm64 runtime-load gates remain pending external evidence in this local report.\n`;
}

async function runCertification(
  argumentsValue: UnifiedSqliteCertificationArguments,
): Promise<Record<string, unknown>> {
  const candidateDirectory = resolveCertifiedCandidateDirectory(
    argumentsValue.dataRoot,
    argumentsValue.candidateTarget,
  );
  if (!existsSync(candidateDirectory)) {
    throw new Error(`Unified SQLite recall staged candidate is missing: ${candidateDirectory}`);
  }
  if (!existsSync(argumentsValue.controlZvecPath)) {
    throw new Error(
      `Unified SQLite recall flat-Zvec control is missing: ${argumentsValue.controlZvecPath}`,
    );
  }
  const { openLegacyV7FlatZvecCertificationControl } =
    await import('./legacy-v7-flat-zvec-certification-control.js');
  const candidateDatabasePath = join(candidateDirectory, 'recall.sqlite');
  const manifestPath = join(candidateDirectory, 'index-manifest.json');
  const activeBefore = snapshotUnifiedSqliteActivePointer(argumentsValue.dataRoot);
  if (
    activeBefore.target &&
    resolve(argumentsValue.dataRoot, activeBefore.target) === candidateDirectory
  ) {
    throw new Error(
      `Unified SQLite recall candidate is active and cannot be certified: ${argumentsValue.candidateTarget}`,
    );
  }
  const manifest = await readRecallIndexManifest(manifestPath);
  if (
    !manifest ||
    manifest.manifestVersion !== 8 ||
    manifest.sqliteRecallDatabase.sqliteVecVersion !== '0.1.9'
  ) {
    throw new Error(
      'Unified SQLite recall candidate manifest must be strict version 8 with sqlite-vec 0.1.9',
    );
  }
  const config = await loadRecallConversationConfig({
    environment: { ...process.env, PI_RECALL_DATA_DIRECTORY: argumentsValue.dataRoot },
  });
  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const projectIdentity = parseProjectIdentity(argumentsValue.projectIdentity);
  const candidate = openSqliteRecallDatabase(candidateDatabasePath, { readOnly: true });
  const control = openLegacyV7FlatZvecCertificationControl({
    databasePath: argumentsValue.controlZvecPath,
    dimensions: config.embeddingStoredDimensions,
  });
  const denseObservations: Array<{
    query: string;
    globalMilliseconds: number[];
    projectMilliseconds: number[];
    globalTopResultMatches: boolean;
    projectTopResultMatches: boolean;
    globalTopEightOverlap: number;
    projectTopEightOverlap: number;
  }> = [];
  try {
    for (const query of DENSE_CERTIFICATION_QUERIES) {
      const embedding = await embeddingProvider.embedQuery(query);
      let controlGlobalIds: string[] = [];
      let controlProjectIds: string[] = [];
      let candidateGlobalIds: string[] = [];
      let candidateProjectIds: string[] = [];
      const globalMilliseconds: number[] = [];
      const projectMilliseconds: number[] = [];
      for (let repetition = 0; repetition < BENCHMARK_REPETITIONS; repetition += 1) {
        controlGlobalIds = control.searchDocumentIds(embedding, 8);
        controlProjectIds = control.searchDocumentIds(embedding, 8, projectIdentity);
        let started = performance.now();
        candidateGlobalIds = candidate.searchDenseCandidates(embedding, 8).map(({ id }) => id);
        const globalElapsed = performance.now() - started;
        started = performance.now();
        candidateProjectIds = candidate
          .searchDenseCandidates(embedding, 8, projectIdentity)
          .map(({ id }) => id);
        const projectElapsed = performance.now() - started;
        if (repetition > 0) {
          globalMilliseconds.push(globalElapsed);
          projectMilliseconds.push(projectElapsed);
        }
      }
      denseObservations.push({
        query,
        globalMilliseconds,
        projectMilliseconds,
        globalTopResultMatches: controlGlobalIds[0] === candidateGlobalIds[0],
        projectTopResultMatches: controlProjectIds[0] === candidateProjectIds[0],
        globalTopEightOverlap: candidateGlobalIds.filter((id) => controlGlobalIds.includes(id))
          .length,
        projectTopEightOverlap: candidateProjectIds.filter((id) => controlProjectIds.includes(id))
          .length,
      });
    }
  } catch (error) {
    candidate.close();
    throw error;
  } finally {
    control.close();
  }
  const { invocationObservations, counts, integrity, identity } = (() => {
    try {
      const observations = INVOCATION_CERTIFICATION_PROBES.map((probe) => {
        const milliseconds: number[] = [];
        let results = candidate.searchInvocations(probe.query, 20);
        for (let repetition = 0; repetition < BENCHMARK_REPETITIONS; repetition += 1) {
          const started = performance.now();
          results = candidate.searchInvocations(probe.query, 20);
          if (repetition > 0) {
            milliseconds.push(performance.now() - started);
          }
        }
        return {
          ...probe,
          milliseconds,
          resultCount: results.length,
          matchedExpectedToolName:
            !('expectedToolName' in probe) ||
            results.some(({ toolName }) => toolName === probe.expectedToolName),
        };
      });
      return {
        invocationObservations: observations,
        counts: candidate.readCounts() satisfies SqliteRecallDatabaseCounts,
        integrity: candidate.checkIntegrity() satisfies SqliteRecallIntegrityDiagnostics,
        identity: candidate.identity,
      };
    } finally {
      candidate.close();
    }
  })();

  const { databaseGenerationRootPath: omittedGenerationRoot, ...ungeneratedConfig } = config;
  void omittedGenerationRoot;
  const sourceService = createRecallConversationService({
    ...ungeneratedConfig,
    sqliteDatabasePath: candidateDatabasePath,
    manifestPath,
    indexMaintenanceStatusPath: join(candidateDirectory, 'index-maintenance-status.json'),
    lockPath: join(candidateDirectory, 'certification-read-only.lock'),
  });
  const sourceObservations: Array<Record<string, unknown>> = [];
  for (const probe of SOURCE_CERTIFICATION_PROBES) {
    const search = await sourceService.searchSource(probe.query, 20, {
      scope: RecallSearchScope.GLOBAL,
    });
    const locations = search.results.map((result) => ({
      sessionPath: result.sessionPath,
      sourceLineStart: result.sourceLineStart,
      sourceLineEnd: result.sourceLineEnd,
      entryId: result.entryId,
      role: readSourceRole(result.sessionPath, result.sourceLineStart),
    }));
    const exactLocation = locations.find(
      (location) => !('requiredRole' in probe) || location.role === probe.requiredRole,
    );
    sourceObservations.push({
      kind: probe.kind,
      query: probe.query,
      filesScanned: search.filesScanned,
      failures: search.failures.length,
      exactLocation: exactLocation ?? null,
      passed: search.failures.length === 0 && exactLocation !== undefined,
    });
  }

  const clone =
    argumentsValue.scratchRoot &&
    argumentsValue.representativeSessionPath &&
    argumentsValue.blockDevice
      ? await (async () => {
          // Argument parsing requires all three clone values together; this branch proves presence.
          const [tokenizer, ignoredPhysicalSessionPathList] = await Promise.all([
            loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory }),
            listIgnoredPhysicalSessionPaths(config.physicalSessionIgnoreStatePath),
          ]);
          const resolveSessionProjectIdentity = createLineageResolver(
            config.projectLineages,
            resolveProjectIdentity,
          );
          return certifyDisposableUnifiedSqliteClone({
            candidateDirectory,
            candidateDatabasePath,
            scratchRoot: argumentsValue.scratchRoot!,
            dataRoot: argumentsValue.dataRoot,
            representativeSessionPath: argumentsValue.representativeSessionPath!,
            blockDevice: argumentsValue.blockDevice!,
            runChangedSessionIndex: (database, onProgress) =>
              indexChangedConversationSessions({
                sessionsDirectory: config.sessionsDirectory,
                database,
                embeddingProvider,
                tokenizer,
                ignoredPhysicalSessionPaths: new Set(ignoredPhysicalSessionPathList),
                chunkPolicy: {
                  maxTokens: manifest.chunkPolicy.maxTokens,
                  overlapTokens: manifest.chunkPolicy.overlapTokens,
                },
                resolveProjectIdentity: resolveSessionProjectIdentity,
                onProgress,
              }),
          });
        })()
      : null;
  const packageAvailability = await Promise.all([
    inspectPublishedSqliteVecPackage('sqlite-vec-darwin-x64'),
    inspectPublishedSqliteVecPackage('sqlite-vec-darwin-arm64'),
  ]);
  const activeAfter = snapshotUnifiedSqliteActivePointer(argumentsValue.dataRoot);
  const candidateInactive =
    JSON.stringify(activeBefore) === JSON.stringify(activeAfter) &&
    (!activeAfter.target ||
      resolve(argumentsValue.dataRoot, activeAfter.target) !== candidateDirectory);
  const storageBytes = readDatabaseAllocatedBytes(candidateDatabasePath);
  const globalTimes = denseObservations.flatMap((item) => item.globalMilliseconds);
  const projectTimes = denseObservations.flatMap((item) => item.projectMilliseconds);
  const invocationTimes = invocationObservations.flatMap(({ milliseconds }) => milliseconds);
  const gates = evaluateUnifiedSqliteCertificationGates({
    storageBytes,
    projectP95Milliseconds: readPercentile(projectTimes, 0.95),
    globalP95Milliseconds: readPercentile(globalTimes, 0.95),
    invocationP95Milliseconds: readPercentile(invocationTimes, 0.95),
    denseTopResultsMatch: denseObservations.map(
      (item) => Boolean(item.globalTopResultMatches) && Boolean(item.projectTopResultMatches),
    ),
    globalTopEightOverlaps: denseObservations.map((item) => Number(item.globalTopEightOverlap)),
    invocationProbePasses: invocationObservations.map(
      (item) => item.resultCount > 0 && item.matchedExpectedToolName,
    ),
    sourceProbePasses: sourceObservations.map((item) => Boolean(item.passed)),
    integrityHealthy: integrity.healthy && integrity.invocationFtsIntegrityChecked === false,
    linuxX64LoadPassed:
      process.platform === 'linux' &&
      process.arch === 'x64' &&
      identity.sqliteVecVersion === 'v0.1.9',
    macOsPackagesAvailable: packageAvailability.every(
      (packageCheck) => packageCheck.packageAvailable === true,
    ),
    macOsX64RuntimeLoadPassed: null,
    macOsArm64RuntimeLoadPassed: null,
    candidateInactive,
    clonePassed: clone ? Boolean(clone.passed) : null,
  });
  const localCandidateGateNames = Object.keys(gates).filter(
    (gate) => gate !== 'macOsX64RuntimeLoad' && gate !== 'macOsArm64RuntimeLoad',
  );
  const candidatePreActivationPassed =
    clone !== null && localCandidateGateNames.every((gate) => gates[gate] === true);
  const releaseReady =
    candidatePreActivationPassed &&
    gates.macOsX64RuntimeLoad === true &&
    gates.macOsArm64RuntimeLoad === true;
  return {
    reportVersion: 1,
    issue: 172,
    measuredAt: new Date().toISOString(),
    candidateTarget: argumentsValue.candidateTarget,
    candidateDirectory,
    controlZvecPath: argumentsValue.controlZvecPath,
    safety: {
      activeBefore,
      activeAfter,
      candidateInactive,
      candidateOpenedReadOnly: true,
      cloneOnlyMutation: true,
    },
    manifest,
    sqliteIdentity: identity,
    linuxX64SqliteVecLoad: {
      executed: process.platform === 'linux' && process.arch === 'x64',
      version: identity.sqliteVecVersion,
      passed:
        process.platform === 'linux' &&
        process.arch === 'x64' &&
        identity.sqliteVecVersion === 'v0.1.9',
    },
    macOsPackageAvailability: packageAvailability,
    macOsRuntimeLoadEvidence: {
      x64: {
        runner: 'macos-26-intel',
        expectedProcessArch: 'x64',
        status: 'pending external GitHub Actions evidence',
      },
      arm64: {
        runner: 'macos-26',
        expectedProcessArch: 'arm64',
        status: 'pending external GitHub Actions evidence',
      },
      evidenceSource: '.github/workflows/sqlite-vec-platform-smoke.yml',
      successfulRunUrl: null,
    },
    counts,
    storage: { allocatedBytes: storageBytes, maximumBytes: MAXIMUM_CERTIFIED_STORAGE_BYTES },
    integrity,
    denseBenchmark: {
      observations: denseObservations,
      globalP95Milliseconds: readPercentile(globalTimes, 0.95),
      projectP95Milliseconds: readPercentile(projectTimes, 0.95),
    },
    invocationBenchmark: {
      observations: invocationObservations,
      p95Milliseconds: readPercentile(invocationTimes, 0.95),
    },
    sourceSearch: { observations: sourceObservations },
    cloneCertification: clone ?? { status: 'pending; rerun with all clone flags' },
    gates,
    candidatePreActivationPassed,
    releaseReady,
    passed: releaseReady,
  };
}

async function main(): Promise<void> {
  const argumentsList = process.argv.slice(2);
  if (runCloneChildMode(argumentsList)) {
    return;
  }
  if (argumentsList.length === 1 && argumentsList[0] === '--help') {
    process.stdout.write(`${USAGE}\n`);
    return;
  }
  const argumentsValue = readUnifiedSqliteCertificationArguments(argumentsList);
  const report = await runCertification(argumentsValue);
  const sanitized = sanitizeUnifiedSqliteCertificationReport(report, {
    [argumentsValue.dataRoot]: '$DATA_ROOT',
    [argumentsValue.controlZvecPath]: '$CONTROL_ZVEC',
    ...(argumentsValue.scratchRoot ? { [argumentsValue.scratchRoot]: '$SCRATCH_ROOT' } : {}),
  });
  if (!isUnknownRecord(sanitized)) {
    throw new Error('Unified SQLite recall certification report sanitization failed');
  }
  process.stdout.write(`${JSON.stringify(sanitized, null, 2)}\n`);
  if (argumentsValue.outputPath) {
    await mkdir(dirname(argumentsValue.outputPath), { recursive: true });
    await writeFile(argumentsValue.outputPath, `${JSON.stringify(sanitized, null, 2)}\n`, 'utf8');
    await writeFile(REPORT_MARKDOWN_PATH, formatCertificationMarkdown(sanitized), 'utf8');
  }
  if (!report.passed) {
    process.exitCode = 2;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
  await main();
}
