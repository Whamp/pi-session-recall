import { createHash } from 'node:crypto';
import {
  mkdirSync,
  openSync,
  closeSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
  fsyncSync,
} from 'node:fs';
import { mkdir, open, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { spawn, spawnSync } from 'node:child_process';
import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecStatus,
} from '@zvec/zvec';

const PROTOTYPE_ROOT = resolve('.prototype-data/incremental-ingestion');
const PROJECTION_COLLECTION_PATH = join(PROTOTYPE_ROOT, 'projection-zvec');
const EVIDENCE_COLLECTION_PATH = join(PROTOTYPE_ROOT, 'evidence-zvec');
const MARKER_SPOOL_PATH = join(PROTOTYPE_ROOT, 'markers');
const SESSION_DIRECTORY = join(homedir(), '.pi', 'agent', 'sessions');
const PROJECTION_PAYLOAD_BYTE_SIZES = [
  1024,
  64 * 1024,
  256 * 1024,
  1024 * 1024,
  4 * 1024 * 1024,
  16 * 1024 * 1024,
];
const EVIDENCE_WRITE_BATCH_SIZES = [1, 8, 32, 128];

interface DistributionSummary {
  count: number;
  minimum: number;
  p50: number;
  p95: number;
  p99: number;
  maximum: number;
}

interface ProjectionPayloadMeasurement {
  payloadBytes: number;
  updateMilliseconds: DistributionSummary;
  fetchMilliseconds: DistributionSummary;
  preservedUntouchedFields: boolean;
}

export interface IncrementalIngestionMeasurementReport {
  projection: {
    payloads: ProjectionPayloadMeasurement[];
    scalarQueryCount: number;
    cleanClosePayloadBefore: { bytes: number; sha256: string } | null;
    cleanClosePayloadAfterReadOnlyReopen: { bytes: number; sha256: string } | null;
    reopenPreservedPayload: boolean;
    abruptExitVisibleBeforeRecovery: boolean;
    abruptExitRecoveredByReadWriteOpen: boolean;
    abruptExitVisibleAfterRecovery: boolean;
    largePayloadReopenStress: {
      payloadBytes: number;
      updatesPerTrial: number;
      trials: number;
      successfulReopens: number;
    };
  };
  immediateBookkeeping: {
    atomicMarkerWithoutFsyncMilliseconds: DistributionSummary;
    durableMarkerWithFsyncMilliseconds: DistributionSummary;
    detachedNodeSpawnCallMilliseconds: DistributionSummary;
    detachedNodeExitMilliseconds: DistributionSummary;
  };
  startupRecovery: {
    sessionFileCount: number;
    totalSessionBytes: number;
    metadataSweepMilliseconds: DistributionSummary;
    globalIndexStateBytes: number;
    globalIndexStateReadParseMilliseconds: DistributionSummary;
  };
  corpusActivity: {
    sampledSessionCount: number;
    sampledSessionBytes: number;
    observedPositiveAppendGaps: number;
    appendGapSeconds: DistributionSummary;
    gapCountsAboveSeconds: Record<string, number>;
  };
  writeWindows: Array<{
    batchSize: number;
    openMilliseconds: DistributionSummary;
    writeMilliseconds: DistributionSummary;
    closeMilliseconds: DistributionSummary;
    fullWindowMilliseconds: DistributionSummary;
  }>;
  freshEvidenceReadOnlyVisibility: {
    fetchCount: number;
    lexicalCount: number;
    denseCount: number;
    indexCompleteness: number;
  };
}

function summarizeDistribution(values: number[]): DistributionSummary {
  if (values.length === 0) {
    return { count: 0, minimum: 0, p50: 0, p95: 0, p99: 0, maximum: 0 };
  }
  const sorted = [...values].sort((left, right) => left - right);
  const percentile = (fraction: number): number => {
    const index = Math.min(Math.ceil(fraction * sorted.length) - 1, sorted.length - 1);
    return sorted[Math.max(index, 0)] ?? 0;
  };
  return {
    count: sorted.length,
    minimum: sorted[0] ?? 0,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    maximum: sorted.at(-1) ?? 0,
  };
}

function assertSuccessfulZvecStatus(status: ZVecStatus | ZVecStatus[], operation: string): void {
  const statuses = Array.isArray(status) ? status : [status];
  const failed = statuses.find((candidate) => !candidate.ok);
  if (failed) {
    throw new Error(`Prototype zvec ${operation} failed: ${failed.code}: ${failed.message}`);
  }
}

function createProjectionCollection(
  collectionPath: string = PROJECTION_COLLECTION_PATH,
): ZVecCollection {
  return ZVecCreateAndOpen(
    collectionPath,
    new ZVecCollectionSchema({
      name: 'prototype_recall_session_projection',
      fields: [
        { name: 'schemaVersion', dataType: ZVecDataType.INT32 },
        { name: 'recordKind', dataType: ZVecDataType.STRING },
        { name: 'physicalSessionId', dataType: ZVecDataType.STRING },
        { name: 'logicalSessionId', dataType: ZVecDataType.STRING },
        { name: 'sessionPathDigest', dataType: ZVecDataType.STRING },
        { name: 'generationId', dataType: ZVecDataType.STRING },
        { name: 'appendByteOffset', dataType: ZVecDataType.INT64 },
        { name: 'boundaryFingerprint', dataType: ZVecDataType.STRING },
        { name: 'markerHighWater', dataType: ZVecDataType.ARRAY_STRING },
        { name: 'repairStatus', dataType: ZVecDataType.STRING },
        { name: 'activeTailState', dataType: ZVecDataType.STRING },
      ],
    }),
  );
}

function createEvidenceCollection(): ZVecCollection {
  return ZVecCreateAndOpen(
    EVIDENCE_COLLECTION_PATH,
    new ZVecCollectionSchema({
      name: 'prototype_recall_evidence',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: 2560,
        indexParams: {
          indexType: ZVecIndexType.HNSW,
          metricType: ZVecMetricType.COSINE,
          m: 50,
          efConstruction: 500,
        },
      },
      fields: [
        { name: 'checksum', dataType: ZVecDataType.STRING },
        {
          name: 'content',
          dataType: ZVecDataType.STRING,
          indexParams: {
            indexType: ZVecIndexType.FTS,
            tokenizerName: 'standard',
            filters: ['lowercase'],
          },
        },
      ],
    }),
  );
}

function createJsonPayload(targetBytes: number): string {
  const fixed = '{"entries":""}'.length;
  return `{"entries":"${'x'.repeat(Math.max(targetBytes - fixed, 0))}"}`;
}

function readFetchedField(
  collection: ZVecCollection,
  documentId: string,
  fieldName: string,
): unknown {
  const document = collection.fetchSync({
    ids: documentId,
    outputFields: [fieldName],
    includeVector: false,
  })[documentId];
  return document?.fields[fieldName];
}

function summarizeStringField(value: unknown): { bytes: number; sha256: string } | null {
  if (typeof value !== 'string') {
    return null;
  }
  return {
    bytes: Buffer.byteLength(value),
    sha256: createHash('sha256').update(value).digest('hex'),
  };
}

function runLargeProjectionReopenStress(): {
  payloadBytes: number;
  updatesPerTrial: number;
  trials: number;
  successfulReopens: number;
} {
  const payload = createJsonPayload(16 * 1024 * 1024);
  const trials = 10;
  const updatesPerTrial = 5;
  let successfulReopens = 0;
  for (let trial = 0; trial < trials; trial += 1) {
    const collectionPath = join(PROTOTYPE_ROOT, `projection-stress-${trial}`);
    const collection = createProjectionCollection(collectionPath);
    assertSuccessfulZvecStatus(
      collection.upsertSync({
        id: 'logical_stress_document',
        fields: {
          schemaVersion: 1,
          recordKind: 'logical',
          physicalSessionId: 'physical_stress',
          logicalSessionId: 'logical_stress',
          sessionPathDigest: 'stress_path_digest',
          generationId: 'generation_1',
          appendByteOffset: 0,
          boundaryFingerprint: '',
          markerHighWater: [],
          repairStatus: 'healthy',
          activeTailState: '{}',
        },
      }),
      'large projection stress insert',
    );
    for (let update = 0; update < updatesPerTrial; update += 1) {
      assertSuccessfulZvecStatus(
        collection.updateSync({
          id: 'logical_stress_document',
          fields: { activeTailState: payload },
        }),
        'large projection stress update',
      );
    }
    collection.closeSync();
    const reopened = ZVecOpen(collectionPath, { readOnly: true });
    const value = readFetchedField(reopened, 'logical_stress_document', 'activeTailState');
    if (value === payload) {
      successfulReopens += 1;
    }
    reopened.closeSync();
    rmSync(collectionPath, { recursive: true, force: true });
  }
  return {
    payloadBytes: Buffer.byteLength(payload),
    updatesPerTrial,
    trials,
    successfulReopens,
  };
}

function runProjectionMeasurement(): IncrementalIngestionMeasurementReport['projection'] {
  const collection = createProjectionCollection();
  assertSuccessfulZvecStatus(
    collection.upsertSync([
      {
        id: 'physical_prototype_document',
        fields: {
          schemaVersion: 1,
          recordKind: 'physical',
          physicalSessionId: 'physical_prototype',
          logicalSessionId: '',
          sessionPathDigest: createHash('sha256').update('prototype-path').digest('hex'),
          generationId: 'generation_1',
          appendByteOffset: 4096,
          boundaryFingerprint: 'boundary_1',
          markerHighWater: [],
          repairStatus: 'healthy',
          activeTailState: '{}',
        },
      },
      {
        id: 'logical_prototype_document',
        fields: {
          schemaVersion: 1,
          recordKind: 'logical',
          physicalSessionId: 'physical_prototype',
          logicalSessionId: 'logical_prototype',
          sessionPathDigest: createHash('sha256').update('prototype-path').digest('hex'),
          generationId: 'generation_1',
          appendByteOffset: 0,
          boundaryFingerprint: '',
          markerHighWater: ['runtime_a:1'],
          repairStatus: 'healthy',
          activeTailState: '{}',
        },
      },
    ]),
    'initial projection upsert',
  );

  const payloads = PROJECTION_PAYLOAD_BYTE_SIZES.map((payloadBytes) => {
    const payload = createJsonPayload(payloadBytes);
    const updateDurations: number[] = [];
    const fetchDurations: number[] = [];
    let preservedUntouchedFields = true;
    for (let iteration = 0; iteration < 10; iteration += 1) {
      const updateStartedAt = performance.now();
      assertSuccessfulZvecStatus(
        collection.updateSync({
          id: 'logical_prototype_document',
          fields: {
            activeTailState: payload,
            markerHighWater: [`runtime_a:${iteration + 2}`],
          },
        }),
        `projection update ${payloadBytes}`,
      );
      updateDurations.push(performance.now() - updateStartedAt);
      const fetchStartedAt = performance.now();
      const documents = collection.fetchSync({
        ids: 'logical_prototype_document',
        outputFields: ['activeTailState', 'logicalSessionId', 'repairStatus'],
        includeVector: false,
      });
      fetchDurations.push(performance.now() - fetchStartedAt);
      const fields = documents.logical_prototype_document?.fields;
      preservedUntouchedFields =
        preservedUntouchedFields &&
        fields?.activeTailState === payload &&
        fields.logicalSessionId === 'logical_prototype' &&
        fields.repairStatus === 'healthy';
    }
    return {
      payloadBytes: Buffer.byteLength(payload),
      updateMilliseconds: summarizeDistribution(updateDurations),
      fetchMilliseconds: summarizeDistribution(fetchDurations),
      preservedUntouchedFields,
    };
  });
  const scalarQueryCount = collection.querySync({
    filter: "recordKind = 'logical'",
    topk: 10,
    outputFields: ['logicalSessionId'],
    includeVector: false,
  }).length;
  const expectedPayload = readFetchedField(
    collection,
    'logical_prototype_document',
    'activeTailState',
  );
  const cleanClosePayloadBefore = summarizeStringField(expectedPayload);
  collection.closeSync();

  const reopened = ZVecOpen(PROJECTION_COLLECTION_PATH, { readOnly: true });
  const reopenedPayload = readFetchedField(
    reopened,
    'logical_prototype_document',
    'activeTailState',
  );
  const cleanClosePayloadAfterReadOnlyReopen = summarizeStringField(reopenedPayload);
  const reopenPreservedPayload =
    cleanClosePayloadBefore !== null &&
    cleanClosePayloadBefore.sha256 === cleanClosePayloadAfterReadOnlyReopen?.sha256 &&
    cleanClosePayloadBefore.bytes === cleanClosePayloadAfterReadOnlyReopen.bytes;
  reopened.closeSync();

  const childResult = spawnSync(
    process.execPath,
    [
      '--import',
      'tsx',
      new URL(import.meta.url).pathname,
      '--projection-crash-child',
      PROJECTION_COLLECTION_PATH,
    ],
    { encoding: 'utf8', timeout: 30_000 },
  );
  if (childResult.status !== 0) {
    throw new Error(
      `Prototype abrupt-exit child failed: ${childResult.stderr || childResult.stdout}`,
    );
  }
  const afterCrashReadOnly = ZVecOpen(PROJECTION_COLLECTION_PATH, { readOnly: true });
  const abruptExitVisibleBeforeRecovery =
    readFetchedField(afterCrashReadOnly, 'logical_crash_document', 'logicalSessionId') ===
    'logical_crash';
  afterCrashReadOnly.closeSync();
  const crashRecoveryWriter = ZVecOpen(PROJECTION_COLLECTION_PATH);
  const abruptExitRecoveredByReadWriteOpen =
    readFetchedField(crashRecoveryWriter, 'logical_crash_document', 'logicalSessionId') ===
    'logical_crash';
  crashRecoveryWriter.closeSync();
  const afterCrashRecovery = ZVecOpen(PROJECTION_COLLECTION_PATH, { readOnly: true });
  const abruptExitVisibleAfterRecovery =
    readFetchedField(afterCrashRecovery, 'logical_crash_document', 'logicalSessionId') ===
    'logical_crash';
  afterCrashRecovery.closeSync();

  return {
    payloads,
    scalarQueryCount,
    cleanClosePayloadBefore,
    cleanClosePayloadAfterReadOnlyReopen,
    reopenPreservedPayload,
    abruptExitVisibleBeforeRecovery,
    abruptExitRecoveredByReadWriteOpen,
    abruptExitVisibleAfterRecovery,
    largePayloadReopenStress: runLargeProjectionReopenStress(),
  };
}

function runProjectionCrashChild(collectionPath: string): never {
  const collection = ZVecOpen(collectionPath);
  assertSuccessfulZvecStatus(
    collection.upsertSync({
      id: 'logical_crash_document',
      fields: {
        schemaVersion: 1,
        recordKind: 'logical',
        physicalSessionId: 'physical_crash',
        logicalSessionId: 'logical_crash',
        sessionPathDigest: 'crash_path_digest',
        generationId: 'generation_1',
        appendByteOffset: 0,
        boundaryFingerprint: '',
        markerHighWater: ['runtime_crash:1'],
        repairStatus: 'healthy',
        activeTailState: '{"entries":[]}',
      },
    }),
    'abrupt-exit projection upsert',
  );
  process.exit(0);
}

function writeAtomicMarker(path: string, content: string, durable: boolean): void {
  const temporaryPath = `${path}.tmp`;
  writeFileSync(temporaryPath, content, 'utf8');
  if (durable) {
    const fileDescriptor = openSync(temporaryPath, 'r');
    fsyncSync(fileDescriptor);
    closeSync(fileDescriptor);
  }
  renameSync(temporaryPath, path);
  if (durable) {
    const directoryDescriptor = openSync(dirname(path), 'r');
    fsyncSync(directoryDescriptor);
    closeSync(directoryDescriptor);
  }
}

async function runImmediateBookkeepingMeasurement(): Promise<
  IncrementalIngestionMeasurementReport['immediateBookkeeping']
> {
  mkdirSync(MARKER_SPOOL_PATH, { recursive: true });
  const content = `${JSON.stringify({
    version: 1,
    markerId: 'prototype_marker',
    sessionPathDigest: 'prototype_path_digest',
    trigger: 'activity',
    runtimeId: 'runtime_a',
    runtimeSequence: 1,
  })}\n`;
  const withoutFsync: number[] = [];
  const withFsync: number[] = [];
  for (let iteration = 0; iteration < 50; iteration += 1) {
    const fastStartedAt = performance.now();
    writeAtomicMarker(join(MARKER_SPOOL_PATH, `fast-${iteration}.json`), content, false);
    withoutFsync.push(performance.now() - fastStartedAt);
    const durableStartedAt = performance.now();
    writeAtomicMarker(join(MARKER_SPOOL_PATH, `durable-${iteration}.json`), content, true);
    withFsync.push(performance.now() - durableStartedAt);
  }
  const spawnCalls: number[] = [];
  const childExits: number[] = [];
  for (let iteration = 0; iteration < 20; iteration += 1) {
    const startedAt = performance.now();
    const child = spawn(process.execPath, ['-e', 'process.exit(0)'], {
      detached: true,
      stdio: 'ignore',
    });
    spawnCalls.push(performance.now() - startedAt);
    await new Promise<void>((resolveChild, rejectChild) => {
      child.once('error', rejectChild);
      child.once('exit', (code) => {
        childExits.push(performance.now() - startedAt);
        if (code === 0) {
          resolveChild();
        } else {
          rejectChild(new Error(`Prototype detached Node child exited with code ${code}`));
        }
      });
    });
  }
  return {
    atomicMarkerWithoutFsyncMilliseconds: summarizeDistribution(withoutFsync),
    durableMarkerWithFsyncMilliseconds: summarizeDistribution(withFsync),
    detachedNodeSpawnCallMilliseconds: summarizeDistribution(spawnCalls),
    detachedNodeExitMilliseconds: summarizeDistribution(childExits),
  };
}

async function listSessionFileMetadata(): Promise<Array<{ size: number; mtimeMs: number }>> {
  const files: Array<{ size: number; mtimeMs: number }> = [];
  async function visit(directory: string): Promise<void> {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        const fileStats = await stat(path);
        files.push({ size: fileStats.size, mtimeMs: fileStats.mtimeMs });
      }
    }
  }
  await visit(SESSION_DIRECTORY);
  return files;
}

async function runStartupRecoveryMeasurement(): Promise<
  IncrementalIngestionMeasurementReport['startupRecovery']
> {
  const sweepDurations: number[] = [];
  let metadata: Array<{ size: number; mtimeMs: number }> = [];
  for (let iteration = 0; iteration < 5; iteration += 1) {
    const startedAt = performance.now();
    metadata = await listSessionFileMetadata();
    sweepDurations.push(performance.now() - startedAt);
  }
  const statePath = join(homedir(), '.pi', 'agent', 'recall', 'index-state.json');
  const stateStats = await stat(statePath);
  const parseDurations: number[] = [];
  for (let iteration = 0; iteration < 5; iteration += 1) {
    const startedAt = performance.now();
    JSON.parse(await readFile(statePath, 'utf8')) as unknown;
    parseDurations.push(performance.now() - startedAt);
  }
  return {
    sessionFileCount: metadata.length,
    totalSessionBytes: metadata.reduce((sum, item) => sum + item.size, 0),
    metadataSweepMilliseconds: summarizeDistribution(sweepDurations),
    globalIndexStateBytes: stateStats.size,
    globalIndexStateReadParseMilliseconds: summarizeDistribution(parseDurations),
  };
}

function listRecentSessionFiles(): Array<{ path: string; size: number; mtimeMs: number }> {
  const files: Array<{ path: string; size: number; mtimeMs: number }> = [];
  const visit = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        const fileStats = statSync(path);
        files.push({ path, size: fileStats.size, mtimeMs: fileStats.mtimeMs });
      }
    }
  };
  visit(SESSION_DIRECTORY);
  return files.sort((left, right) => right.mtimeMs - left.mtimeMs);
}

function readRecordTimestampMilliseconds(record: unknown): number | undefined {
  if (!record || typeof record !== 'object') {
    return undefined;
  }
  const timestamp = Reflect.get(record, 'timestamp');
  if (typeof timestamp === 'string') {
    const parsed = Date.parse(timestamp);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function runCorpusActivityMeasurement(): IncrementalIngestionMeasurementReport['corpusActivity'] {
  const maximumSampleBytes = 100 * 1024 * 1024;
  const maximumSessionCount = 50;
  const selected: Array<{ path: string; size: number }> = [];
  let sampledBytes = 0;
  for (const file of listRecentSessionFiles()) {
    if (selected.length >= maximumSessionCount || sampledBytes + file.size > maximumSampleBytes) {
      continue;
    }
    selected.push(file);
    sampledBytes += file.size;
  }
  const gaps: number[] = [];
  for (const file of selected) {
    let previousTimestamp: number | undefined;
    for (const line of readFileSync(file.path, 'utf8').split('\n')) {
      if (!line.trim()) {
        continue;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        continue;
      }
      const timestamp = readRecordTimestampMilliseconds(parsed);
      if (
        timestamp !== undefined &&
        previousTimestamp !== undefined &&
        timestamp > previousTimestamp
      ) {
        gaps.push((timestamp - previousTimestamp) / 1000);
      }
      if (timestamp !== undefined) {
        previousTimestamp = timestamp;
      }
    }
  }
  const gapThresholds = [60, 120, 300, 600, 1800];
  return {
    sampledSessionCount: selected.length,
    sampledSessionBytes: sampledBytes,
    observedPositiveAppendGaps: gaps.length,
    appendGapSeconds: summarizeDistribution(gaps),
    gapCountsAboveSeconds: Object.fromEntries(
      gapThresholds.map((threshold) => [
        String(threshold),
        gaps.filter((gap) => gap >= threshold).length,
      ]),
    ),
  };
}

function createPrototypeEmbedding(seed: number): number[] {
  const vector = Array.from({ length: 2560 }, (_, index) => Math.sin(seed * 31 + index));
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return vector.map((value) => value / norm);
}

function readFreshEvidenceVisibility(): IncrementalIngestionMeasurementReport['freshEvidenceReadOnlyVisibility'] {
  const collection = ZVecOpen(EVIDENCE_COLLECTION_PATH, { readOnly: true });
  const documentId = 'batch-128-document-0';
  const fetchCount = collection.fetchSync({
    ids: documentId,
    outputFields: ['content'],
    includeVector: false,
  })[documentId]
    ? 1
    : 0;
  const lexicalCount = collection.querySync({
    fieldName: 'content',
    fts: { matchString: 'prototype' },
    topk: 5,
    outputFields: ['content'],
    includeVector: false,
    params: { indexType: ZVecIndexType.FTS, defaultOperator: 'OR' },
  }).length;
  const denseCount = collection.querySync({
    fieldName: 'embedding',
    vector: createPrototypeEmbedding(1),
    topk: 5,
    outputFields: ['content'],
    includeVector: false,
    params: { indexType: ZVecIndexType.HNSW, ef: 300 },
  }).length;
  const indexCompleteness = collection.stats.indexCompleteness.embedding ?? 0;
  collection.closeSync();
  return { fetchCount, lexicalCount, denseCount, indexCompleteness };
}

function runWriteWindowMeasurement(): IncrementalIngestionMeasurementReport['writeWindows'] {
  const initial = createEvidenceCollection();
  initial.closeSync();
  return EVIDENCE_WRITE_BATCH_SIZES.map((batchSize) => {
    const openDurations: number[] = [];
    const writeDurations: number[] = [];
    const closeDurations: number[] = [];
    const fullDurations: number[] = [];
    for (let iteration = 0; iteration < 10; iteration += 1) {
      const windowStartedAt = performance.now();
      const openStartedAt = performance.now();
      const collection = ZVecOpen(EVIDENCE_COLLECTION_PATH);
      openDurations.push(performance.now() - openStartedAt);
      const documents = Array.from({ length: batchSize }, (_, index) => {
        const id = `batch-${batchSize}-document-${index}`;
        return {
          id,
          vectors: { embedding: createPrototypeEmbedding(index + 1) },
          fields: {
            checksum: createHash('sha256').update(id).digest('hex'),
            content: `prototype evidence ${index} ${'x'.repeat(1024)}`,
          },
        };
      });
      const writeStartedAt = performance.now();
      assertSuccessfulZvecStatus(collection.upsertSync(documents), `evidence batch ${batchSize}`);
      writeDurations.push(performance.now() - writeStartedAt);
      const closeStartedAt = performance.now();
      collection.closeSync();
      closeDurations.push(performance.now() - closeStartedAt);
      fullDurations.push(performance.now() - windowStartedAt);
    }
    return {
      batchSize,
      openMilliseconds: summarizeDistribution(openDurations),
      writeMilliseconds: summarizeDistribution(writeDurations),
      closeMilliseconds: summarizeDistribution(closeDurations),
      fullWindowMilliseconds: summarizeDistribution(fullDurations),
    };
  });
}

/** Runs scratch-only persistence, marker, sweep, activity, and write-window measurements. */
export async function measureIncrementalIngestionPrototype(): Promise<IncrementalIngestionMeasurementReport> {
  rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  mkdirSync(PROTOTYPE_ROOT, { recursive: true });
  try {
    const projection = runProjectionMeasurement();
    const immediateBookkeeping = await runImmediateBookkeepingMeasurement();
    const startupRecovery = await runStartupRecoveryMeasurement();
    const corpusActivity = runCorpusActivityMeasurement();
    const writeWindows = runWriteWindowMeasurement();
    const freshEvidenceReadOnlyVisibility = readFreshEvidenceVisibility();
    return {
      projection,
      immediateBookkeeping,
      startupRecovery,
      corpusActivity,
      writeWindows,
      freshEvidenceReadOnlyVisibility,
    };
  } finally {
    rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  }
}

if (process.argv[2] === '--projection-crash-child') {
  const collectionPath = process.argv[3];
  if (!collectionPath) {
    throw new Error('Prototype abrupt-exit child missing collection path');
  }
  runProjectionCrashChild(collectionPath);
}
