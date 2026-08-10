import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, opendirSync, readFileSync, statSync } from 'node:fs';
import { mkdir, readlink, stat, utimes, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { createInterface } from 'node:readline';
import { performance } from 'node:perf_hooks';
import { setTimeout as sleep } from 'node:timers/promises';

import { RecallSearchScope } from './enums.js';
import { openDenseRecallConversationStore } from './dense-recall-conversation-store.js';
import { createOctenHttpEmbeddingProvider } from './octen-http-embedding-provider.js';
import { listIgnoredPhysicalSessionPaths } from './physical-session-ignore.js';
import { openRecallCatalog, type RecallCatalogSessionState } from './openRecallCatalog.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { resolveActiveRecallDatabasePaths } from './recall-database-generation.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';

const GIBIBYTE = 1_024 ** 3;
const MEBIBYTE = 1_024 ** 2;
const MAXIMUM_COMPACT_STORAGE_BYTES = 2.5 * GIBIBYTE;
const MAXIMUM_DENSE_P95_MILLISECONDS = 100;
const MAXIMUM_INVOCATION_P95_MILLISECONDS = 5;
const MAXIMUM_CHANGED_SESSION_DEVICE_WRITES = 10 * MEBIBYTE;
const BENCHMARK_REPETITIONS = 6;

const DENSE_BENCHMARK_QUERIES = [
  'Why have recent pi-session-recall optimization attempts failed?',
  'How is automatic recall indexing scheduled?',
  'Which corrupted February session files are ignored?',
  'How large is the recall database?',
  'Why would an agent use pi-session-recall instead of searching raw JSONL?',
] as const;

const INVOCATION_BENCHMARK_PROBES = [
  { kind: 'tool-name', query: 'brain_query', expectedToolName: 'brain_query' },
  { kind: 'path', query: '/home/will/.pi/agent/TAILNET.md' },
  { kind: 'url', query: 'http://192.168.0.67:8090/v1' },
  { kind: 'command', query: 'psr optimize' },
  { kind: 'issue-number', query: 'gh issue view 165' },
  { kind: 'flag', query: '--optimize-daily' },
] as const;

const SOURCE_SEARCH_PROBES = [
  { kind: 'result-only-error', query: 'FtsRocksdbReducer', requiredRole: 'toolResult' },
  { kind: 'hardware-identifier', query: 'CT1000P3PSSD8', requiredRole: 'toolResult' },
  { kind: 'filename', query: '2026-02-02T18-31-25' },
  {
    kind: 'command-output',
    query: 'pi - AI coding assistant with read, bash, edit, write tools',
    requiredRole: 'bashExecution',
  },
] as const;

interface CertificationArguments {
  candidateTarget: string;
  blockDevice: string;
  changedSessionPath: string;
  outputPath: string;
}

interface ProcessIoCounters {
  readBytes: number;
  writeBytes: number;
}

interface SourceLocation {
  sessionPath: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  entryId: string | null;
  role: string | null;
}

function readCertificationArguments(argumentsList: readonly string[]): CertificationArguments {
  let candidateTarget = '';
  let blockDevice = '';
  let changedSessionPath = '';
  let outputPath = '';
  for (let index = 0; index < argumentsList.length; index += 2) {
    const flag = argumentsList[index];
    const value = argumentsList[index + 1] ?? '';
    if (!value) {
      throw new Error(
        'Compact recall production certification usage: --candidate <database-target> --block-device <name> --changed-session <path> --output <path>',
      );
    }
    if (flag === '--candidate' && !candidateTarget) {
      candidateTarget = value;
    } else if (flag === '--block-device' && !blockDevice) {
      blockDevice = value;
    } else if (flag === '--changed-session' && !changedSessionPath) {
      changedSessionPath = resolve(value);
    } else if (flag === '--output' && !outputPath) {
      outputPath = value;
    } else {
      throw new Error(
        'Compact recall production certification usage: --candidate <database-target> --block-device <name> --changed-session <path> --output <path>',
      );
    }
  }
  if (!candidateTarget || !blockDevice || !changedSessionPath || !outputPath) {
    throw new Error(
      'Compact recall production certification usage: --candidate <database-target> --block-device <name> --changed-session <path> --output <path>',
    );
  }
  return {
    candidateTarget,
    blockDevice,
    changedSessionPath,
    outputPath: resolve(outputPath),
  };
}

function readDirectoryAllocatedBytes(rootPath: string): number {
  if (!existsSync(rootPath)) {
    return 0;
  }
  let allocatedBytes = 0;
  const pendingPaths = [rootPath];
  while (pendingPaths.length > 0) {
    const currentPath = pendingPaths.pop();
    if (!currentPath) {
      continue;
    }
    const directory = opendirSync(currentPath);
    try {
      while (true) {
        const entry = directory.readSync();
        if (!entry) {
          break;
        }
        const entryPath = join(currentPath, entry.name);
        if (entry.isDirectory()) {
          pendingPaths.push(entryPath);
        } else if (entry.isFile()) {
          allocatedBytes += statSync(entryPath).blocks * 512;
        }
      }
    } finally {
      directory.closeSync();
    }
  }
  return allocatedBytes;
}

function readFileAllocatedBytes(filePath: string): number {
  return existsSync(filePath) ? statSync(filePath).blocks * 512 : 0;
}

function readLatencyPercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    throw new Error('Compact recall production certification latency sample is empty');
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(percentile * sorted.length) - 1)] ?? 0;
}

function readProcessIoCounters(): ProcessIoCounters {
  const counters = new Map<string, number>();
  for (const line of readFileSync('/proc/self/io', 'utf8').split('\n')) {
    const separator = line.indexOf(':');
    if (separator >= 0) {
      counters.set(line.slice(0, separator), Number(line.slice(separator + 1).trim()));
    }
  }
  return {
    readBytes: counters.get('read_bytes') ?? 0,
    writeBytes: counters.get('write_bytes') ?? 0,
  };
}

function readBlockDeviceWrittenBytes(blockDevice: string): number {
  const statPath = `/sys/class/block/${blockDevice}/stat`;
  const fields = readFileSync(statPath, 'utf8').trim().split(/\s+/u);
  const sectorsWritten = Number(fields[6]);
  if (!Number.isFinite(sectorsWritten)) {
    throw new Error(
      `Compact recall production certification block statistics invalid at ${statPath}`,
    );
  }
  return sectorsWritten * 512;
}

async function measureIdleBlockDeviceWrites(
  blockDevice: string,
  durationMilliseconds: number,
): Promise<number> {
  execFileSync('sync');
  const writtenBefore = readBlockDeviceWrittenBytes(blockDevice);
  await sleep(durationMilliseconds);
  execFileSync('sync');
  return readBlockDeviceWrittenBytes(blockDevice) - writtenBefore;
}

function resolveCandidateDirectory(
  dataDirectory: string,
  generationRootPath: string,
  candidateTarget: string,
): string {
  const candidateDirectory = resolve(dataDirectory, candidateTarget);
  const relativeGenerationPath = relative(resolve(generationRootPath), candidateDirectory);
  if (
    relativeGenerationPath.startsWith(`..${sep}`) ||
    relativeGenerationPath === '..' ||
    dirname(candidateDirectory) !== resolve(generationRootPath) ||
    !relativeGenerationPath.startsWith('generation-') ||
    relative(dataDirectory, candidateDirectory) !== candidateTarget
  ) {
    throw new Error(
      `Compact recall production certification candidate target invalid: ${candidateTarget}`,
    );
  }
  return candidateDirectory;
}

function createCandidatePaths(candidateDirectory: string) {
  return {
    databasePath: join(candidateDirectory, 'zvec'),
    catalogPath: join(candidateDirectory, 'recall-catalog.sqlite'),
    statePath: join(candidateDirectory, 'index-state.json'),
    manifestPath: join(candidateDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(candidateDirectory, 'index-maintenance-status.json'),
  };
}

async function readSessionEntryRole(
  sessionPath: string,
  sourceLine: number,
): Promise<string | null> {
  const lines = createInterface({ input: createReadStream(sessionPath), crlfDelay: Infinity });
  let lineNumber = 0;
  try {
    for await (const line of lines) {
      lineNumber += 1;
      if (lineNumber !== sourceLine) {
        continue;
      }
      const entry: unknown = JSON.parse(line);
      if (!entry || typeof entry !== 'object' || !('message' in entry)) {
        return null;
      }
      const message = Reflect.get(entry, 'message');
      return message &&
        typeof message === 'object' &&
        typeof Reflect.get(message, 'role') === 'string'
        ? String(Reflect.get(message, 'role'))
        : null;
    }
  } finally {
    lines.close();
  }
  return null;
}

function snapshotCatalogSessionStates(catalogPath: string): Map<string, RecallCatalogSessionState> {
  const catalog = openRecallCatalog(catalogPath, { readOnly: true });
  try {
    return new Map(
      catalog.listPhysicalSessionPaths().map((sessionPath) => {
        const state = catalog.readPhysicalSessionState(sessionPath);
        if (!state) {
          throw new Error(`Compact recall catalog state disappeared for ${sessionPath}`);
        }
        return [sessionPath, state];
      }),
    );
  } finally {
    catalog.close();
  }
}

async function findExpectedChangedSessionPaths(
  states: ReadonlyMap<string, RecallCatalogSessionState>,
): Promise<Set<string>> {
  const changedPaths = new Set<string>();
  for (const [sessionPath, state] of states) {
    try {
      const source = await stat(sessionPath);
      if (source.size !== state.size || source.mtimeMs !== state.mtimeMs) {
        changedPaths.add(sessionPath);
      }
    } catch {
      changedPaths.add(sessionPath);
    }
  }
  return changedPaths;
}

function statesAreEqual(
  left: RecallCatalogSessionState | undefined,
  right: RecallCatalogSessionState | undefined,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

async function main(): Promise<void> {
  const argumentsValue = readCertificationArguments(process.argv.slice(2));
  const productionConfig = await loadRecallConversationConfig();
  if (!productionConfig.databaseGenerationRootPath) {
    throw new Error('Compact recall production certification requires database generations');
  }
  const dataDirectory = dirname(productionConfig.databaseGenerationRootPath);
  const candidateDirectory = resolveCandidateDirectory(
    dataDirectory,
    productionConfig.databaseGenerationRootPath,
    argumentsValue.candidateTarget,
  );
  const candidatePaths = createCandidatePaths(candidateDirectory);
  const {
    databaseGenerationRootPath: omittedProductionGenerationRootPath,
    ...configWithoutDatabaseGenerations
  } = productionConfig;
  void omittedProductionGenerationRootPath;
  const candidateConfig = {
    ...configWithoutDatabaseGenerations,
    ...candidatePaths,
    lockPath: join(candidateDirectory, 'certification.lock'),
  };
  const activePaths = await resolveActiveRecallDatabasePaths(productionConfig);
  const activeTargetBeforeCertification = existsSync(join(dataDirectory, 'active'))
    ? await readlink(join(dataDirectory, 'active'))
    : null;

  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: productionConfig.embeddingBaseUrl,
    model: productionConfig.embeddingModel,
    nativeDimensions: productionConfig.embeddingNativeDimensions,
    storedDimensions: productionConfig.embeddingStoredDimensions,
    batchSize: productionConfig.embeddingBatchSize,
  });
  const activeControlStore = openZvecConversationStore({
    databasePath: activePaths.databasePath,
    dimensions: productionConfig.embeddingStoredDimensions,
    createIfMissing: false,
    readOnly: true,
  });
  const candidateStore = openDenseRecallConversationStore({
    databasePath: candidatePaths.databasePath,
    createIfMissing: false,
    readOnly: true,
  });
  const denseObservations = [];
  try {
    for (const query of DENSE_BENCHMARK_QUERIES) {
      const embedding = await embeddingProvider.embedQuery(query);
      const controlTimes: number[] = [];
      const candidateTimes: number[] = [];
      let controlIds: string[] = [];
      let candidateIds: string[] = [];
      for (let repetition = 0; repetition < BENCHMARK_REPETITIONS; repetition += 1) {
        let startedAt = performance.now();
        const controlResults = activeControlStore.searchDenseCandidates(embedding, 8);
        const controlElapsed = performance.now() - startedAt;
        startedAt = performance.now();
        const candidateResults = candidateStore.searchDenseCandidates(embedding, 8);
        const candidateElapsed = performance.now() - startedAt;
        if (repetition > 0) {
          controlTimes.push(controlElapsed);
          candidateTimes.push(candidateElapsed);
        }
        controlIds = controlResults.map((result) => result.id);
        candidateIds = candidateResults.map((result) => result.id);
      }
      denseObservations.push({
        query,
        controlHnswMilliseconds: controlTimes,
        candidateFlatMilliseconds: candidateTimes,
        topResultMatches: controlIds[0] === candidateIds[0],
        topEightOverlap: candidateIds.filter((id) => controlIds.includes(id)).length,
      });
    }
  } finally {
    activeControlStore.close();
    candidateStore.close();
  }
  const denseCandidateTimes = denseObservations.flatMap(
    (observation) => observation.candidateFlatMilliseconds,
  );
  const denseP95Milliseconds = readLatencyPercentile(denseCandidateTimes, 0.95);

  const invocationCatalog = openRecallCatalog(candidatePaths.catalogPath, { readOnly: true });
  const invocationObservations = [];
  try {
    for (const probe of INVOCATION_BENCHMARK_PROBES) {
      const milliseconds: number[] = [];
      let resultCount = 0;
      let matchedExpectedToolName = true;
      for (let repetition = 0; repetition < BENCHMARK_REPETITIONS; repetition += 1) {
        const startedAt = performance.now();
        const results = invocationCatalog.searchInvocations(probe.query, 20);
        const elapsed = performance.now() - startedAt;
        if (repetition > 0) {
          milliseconds.push(elapsed);
        }
        resultCount = results.length;
        matchedExpectedToolName =
          !('expectedToolName' in probe) ||
          results.some((result) => result.toolName === probe.expectedToolName);
      }
      invocationObservations.push({
        ...probe,
        milliseconds,
        resultCount,
        matchedExpectedToolName,
      });
    }
  } finally {
    invocationCatalog.close();
  }
  const invocationTimes = invocationObservations.flatMap((observation) => observation.milliseconds);
  const invocationP95Milliseconds = readLatencyPercentile(invocationTimes, 0.95);

  const candidateService = createRecallConversationService(candidateConfig);
  const sourceObservations = [];
  for (const probe of SOURCE_SEARCH_PROBES) {
    const startedAt = performance.now();
    const search = await candidateService.searchSource(probe.query, 20, {
      scope: RecallSearchScope.GLOBAL,
    });
    const locations: SourceLocation[] = [];
    for (const result of search.results) {
      locations.push({
        sessionPath: result.sessionPath,
        sourceLineStart: result.sourceLineStart,
        sourceLineEnd: result.sourceLineEnd,
        entryId: result.entryId,
        role: await readSessionEntryRole(result.sessionPath, result.sourceLineStart),
      });
    }
    const matchingLocation = locations.find(
      (location) => !('requiredRole' in probe) || location.role === probe.requiredRole,
    );
    sourceObservations.push({
      kind: probe.kind,
      query: probe.query,
      elapsedMilliseconds: performance.now() - startedAt,
      filesScanned: search.filesScanned,
      failures: search.failures.length,
      matchedExpectedRole: matchingLocation !== undefined,
      location: matchingLocation ?? null,
    });
  }

  const baselineUpdateResult = await candidateService.index();
  const changedSessionStats = await stat(argumentsValue.changedSessionPath);
  const catalogBeforeUpdate = snapshotCatalogSessionStates(candidatePaths.catalogPath);
  const changedSessionState = catalogBeforeUpdate.get(argumentsValue.changedSessionPath);
  if (
    !changedSessionState ||
    changedSessionState.size !== changedSessionStats.size ||
    changedSessionState.mtimeMs !== changedSessionStats.mtimeMs
  ) {
    throw new Error(
      `Compact recall changed-session probe is not synchronized after baseline update: ${argumentsValue.changedSessionPath}`,
    );
  }
  await utimes(
    argumentsValue.changedSessionPath,
    changedSessionStats.atimeMs / 1_000,
    (changedSessionStats.mtimeMs + 1_000) / 1_000,
  );
  const measuredUpdate = await (async () => {
    try {
      const expectedChangedSessionPaths =
        await findExpectedChangedSessionPaths(catalogBeforeUpdate);
      execFileSync('sync');
      const processIoBeforeUpdate = readProcessIoCounters();
      const deviceWrittenBeforeUpdate = readBlockDeviceWrittenBytes(argumentsValue.blockDevice);
      const updateStartedAt = performance.now();
      const updateResult = await candidateService.index();
      const updateElapsedMilliseconds = performance.now() - updateStartedAt;
      const processIoAfterUpdate = readProcessIoCounters();
      execFileSync('sync');
      const deviceWrittenAfterUpdate = readBlockDeviceWrittenBytes(argumentsValue.blockDevice);
      const catalogAfterUpdate = snapshotCatalogSessionStates(candidatePaths.catalogPath);
      return {
        expectedChangedSessionPaths,
        processIoBeforeUpdate,
        deviceWrittenBeforeUpdate,
        updateResult,
        updateElapsedMilliseconds,
        processIoAfterUpdate,
        deviceWrittenAfterUpdate,
        catalogAfterUpdate,
      };
    } finally {
      await utimes(
        argumentsValue.changedSessionPath,
        changedSessionStats.atimeMs / 1_000,
        changedSessionStats.mtimeMs / 1_000,
      );
    }
  })();
  const restoredSourceUpdateResult = await candidateService.index();
  const unrelatedSessionChanges = [...catalogBeforeUpdate].filter(
    ([sessionPath, beforeState]) =>
      !measuredUpdate.expectedChangedSessionPaths.has(sessionPath) &&
      !statesAreEqual(beforeState, measuredUpdate.catalogAfterUpdate.get(sessionPath)),
  );
  const changedSessionGrossDeviceWrites =
    measuredUpdate.deviceWrittenAfterUpdate - measuredUpdate.deviceWrittenBeforeUpdate;
  const idleDeviceWriteSamples: number[] = [];
  for (let sample = 0; sample < 3; sample += 1) {
    idleDeviceWriteSamples.push(
      await measureIdleBlockDeviceWrites(
        argumentsValue.blockDevice,
        measuredUpdate.updateElapsedMilliseconds,
      ),
    );
  }
  const medianIdleDeviceWrites = readLatencyPercentile(idleDeviceWriteSamples, 0.5);
  const changedSessionDeviceWrites = Math.max(
    0,
    changedSessionGrossDeviceWrites - medianIdleDeviceWrites,
  );
  const sourceMtimeRestored =
    (await stat(argumentsValue.changedSessionPath)).mtimeMs === changedSessionStats.mtimeMs;

  const denseStoreBytes = readDirectoryAllocatedBytes(candidatePaths.databasePath);
  const catalogBytes = ['', '-wal', '-shm'].reduce(
    (total, suffix) => total + readFileAllocatedBytes(`${candidatePaths.catalogPath}${suffix}`),
    0,
  );
  const compactStorageBytes = denseStoreBytes + catalogBytes;
  const activeTargetAfterCertification = existsSync(join(dataDirectory, 'active'))
    ? await readlink(join(dataDirectory, 'active'))
    : null;
  const ignoredPhysicalSessionPaths = await listIgnoredPhysicalSessionPaths(
    productionConfig.physicalSessionIgnoreStatePath,
  );
  const report = {
    reportVersion: 1,
    issue: 172,
    measuredAt: new Date().toISOString(),
    candidateTarget: argumentsValue.candidateTarget,
    candidateDirectory,
    production: {
      sessionsDirectory: productionConfig.sessionsDirectory,
      activeDatabasePath: activePaths.databasePath,
      activeTargetBeforeCertification,
      activeTargetAfterCertification,
      activeDatabaseUnchanged: activeTargetBeforeCertification === activeTargetAfterCertification,
      ignoredPhysicalSessionCount: ignoredPhysicalSessionPaths.length,
    },
    storage: {
      denseStoreBytes,
      catalogBytes,
      compactStorageBytes,
      maximumBytes: MAXIMUM_COMPACT_STORAGE_BYTES,
      passed: compactStorageBytes <= MAXIMUM_COMPACT_STORAGE_BYTES,
    },
    denseBenchmark: {
      observations: denseObservations,
      candidateFlatP95Milliseconds: denseP95Milliseconds,
      maximumP95Milliseconds: MAXIMUM_DENSE_P95_MILLISECONDS,
      matchingTopResults: denseObservations.filter((observation) => observation.topResultMatches)
        .length,
      queryCount: denseObservations.length,
      passed:
        denseP95Milliseconds < MAXIMUM_DENSE_P95_MILLISECONDS &&
        denseObservations.every(
          (observation) => observation.topResultMatches && observation.topEightOverlap >= 7,
        ),
    },
    invocationBenchmark: {
      observations: invocationObservations,
      p95Milliseconds: invocationP95Milliseconds,
      maximumP95Milliseconds: MAXIMUM_INVOCATION_P95_MILLISECONDS,
      passed:
        invocationP95Milliseconds < MAXIMUM_INVOCATION_P95_MILLISECONDS &&
        invocationObservations.every(
          (observation) => observation.resultCount > 0 && observation.matchedExpectedToolName,
        ),
    },
    sourceSearch: {
      observations: sourceObservations,
      passed: sourceObservations.every(
        (observation) =>
          observation.failures === 0 &&
          observation.matchedExpectedRole &&
          observation.location !== null,
      ),
    },
    changedSessionUpdate: {
      sessionPath: argumentsValue.changedSessionPath,
      method: 'reversible source mtime change followed by restoration',
      baselineIndexedSessions: baselineUpdateResult.indexSummary.indexedSessions,
      elapsedMilliseconds: measuredUpdate.updateElapsedMilliseconds,
      expectedChangedSessionCount: measuredUpdate.expectedChangedSessionPaths.size,
      indexedSessions: measuredUpdate.updateResult.indexSummary.indexedSessions,
      removedSessions: measuredUpdate.updateResult.indexSummary.removedSessions,
      newlyEmbeddedDocuments: measuredUpdate.updateResult.indexSummary.newlyEmbeddedChunks,
      reusedVectors: measuredUpdate.updateResult.indexSummary.reusedVectors,
      failedSessions: measuredUpdate.updateResult.indexSummary.failedSessions,
      processReadBytes:
        measuredUpdate.processIoAfterUpdate.readBytes -
        measuredUpdate.processIoBeforeUpdate.readBytes,
      processWriteBytes:
        measuredUpdate.processIoAfterUpdate.writeBytes -
        measuredUpdate.processIoBeforeUpdate.writeBytes,
      grossDeviceWrittenBytes: changedSessionGrossDeviceWrites,
      idleDeviceWriteSamples,
      medianIdleDeviceWrittenBytes: medianIdleDeviceWrites,
      deviceWrittenBytes: changedSessionDeviceWrites,
      deviceWriteMeasurement: 'gross NVMe writes minus median of three equal-duration idle windows',
      maximumDeviceWrittenBytes: MAXIMUM_CHANGED_SESSION_DEVICE_WRITES,
      unrelatedSessionChanges: unrelatedSessionChanges.map(([sessionPath]) => sessionPath),
      sourceMtimeRestored,
      restoredCatalogIndexedSessions: restoredSourceUpdateResult.indexSummary.indexedSessions,
      passed:
        baselineUpdateResult.indexSummary.failedSessions.length === 0 &&
        measuredUpdate.expectedChangedSessionPaths.size === 1 &&
        measuredUpdate.updateResult.indexSummary.indexedSessions === 1 &&
        measuredUpdate.updateResult.indexSummary.failedSessions.length === 0 &&
        changedSessionDeviceWrites < MAXIMUM_CHANGED_SESSION_DEVICE_WRITES &&
        unrelatedSessionChanges.length === 0 &&
        sourceMtimeRestored &&
        restoredSourceUpdateResult.indexSummary.indexedSessions === 1,
    },
  };
  const passed =
    report.production.activeDatabaseUnchanged &&
    report.storage.passed &&
    report.denseBenchmark.passed &&
    report.invocationBenchmark.passed &&
    report.sourceSearch.passed &&
    report.changedSessionUpdate.passed;
  const completedReport = { ...report, passed };
  await mkdir(dirname(argumentsValue.outputPath), { recursive: true });
  await writeFile(
    argumentsValue.outputPath,
    `${JSON.stringify(completedReport, null, 2)}\n`,
    'utf8',
  );
  process.stdout.write(`${JSON.stringify(completedReport, null, 2)}\n`);
  if (!passed) {
    process.exitCode = 2;
  }
}

await main();
