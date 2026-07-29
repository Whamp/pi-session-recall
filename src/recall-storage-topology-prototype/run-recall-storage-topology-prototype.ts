/* eslint-disable no-console -- this throwaway terminal UI intentionally renders with console output */

import { createHash } from 'node:crypto';
import {
  chmodSync,
  closeSync,
  cpSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  truncateSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline/promises';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecDoc,
  type ZVecDocInput,
  type ZVecStatus,
} from '@zvec/zvec';

const PROTOTYPE_NAME = 'PROTOTYPE — predictable recall storage topology';
const PROTOTYPE_ROOT = join(tmpdir(), 'pi-session-recall-storage-topology-prototype');
const GENERATION_ID = 'prototype-generation';
const GENERATION_ROOT = join(PROTOTYPE_ROOT, 'generations', GENERATION_ID);
const MEASUREMENTS_PATH = resolve(dirname(fileURLToPath(import.meta.url)), 'MEASUREMENTS.md');
const DEFAULT_SESSION_COUNT = 24;
const DEFAULT_EMBEDDING_DIMENSIONS = 256;
const WRITE_BATCH_SIZE = 256;
const CRASH_ROW_COUNT = 32;
const byteFormatter = new Intl.NumberFormat('en-US');

interface EntryAnchorFixture {
  anchorId: string;
  physicalSessionProjectionId: string;
  logicalSessionId: string;
  entryId: string;
  parentEntryId: string;
  branchPathLeafIds: string[];
  occurrenceIds: string[];
  sourceLineStart: number;
  sourceLineEnd: number;
}

interface EvidenceOccurrenceFixture {
  occurrenceId: string;
  checksum: string;
  physicalSessionProjectionId: string;
  logicalSessionId: string;
  entryId: string;
  evidenceKind: 'conversation' | 'tool_result';
  content: string;
  identifierContent: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  sourceBlockStart: number;
  sourceBlockEnd: number;
  embedding: number[] | null;
}

interface ProjectionFixture {
  projectionId: string;
  projectionKind: 'physical' | 'logical';
  physicalSessionProjectionId: string;
  logicalSessionId: string;
  revision: number;
  projectionJson: string;
}

interface RepresentativeFixture {
  entryAnchors: EntryAnchorFixture[];
  evidenceOccurrences: EvidenceOccurrenceFixture[];
  projections: ProjectionFixture[];
  mainBranchEndpointEntryId: string;
  abandonedBranchEndpointEntryId: string;
  sharedAncestorOccurrenceId: string;
  mainOnlyOccurrenceId: string;
}

interface ComponentSize {
  files: number;
  apparentBytes: number;
  allocatedBytes: number;
}

interface StoreSizes {
  lexicalSource: ComponentSize;
  dense: ComponentSize;
  projections: ComponentSize;
  generation: ComponentSize;
}

interface ReplayMeasurement {
  strategy: 'blind_upsert' | 'fetch_verify_insert';
  logicalRows: number;
  successfulPhysicalWriteVersions: number;
  sizesAfterPasses: number[];
  sizeAfterOptimize: number;
}

interface RecoveryFaultMeasurement {
  variant: 'intact' | 'truncated' | 'flipped' | 'missing' | 'unreadable';
  opened: boolean;
  logicalRows: number | null;
  fetchedRows: number | null;
  error: string | null;
}

interface NeighborhoodMeasurement {
  selectedEndpointEntryId: string;
  entryIds: string[];
  occurrenceIds: string[];
  stayedInLogicalSession: boolean;
  unrelatedBranchRejected: boolean;
}

interface DeletionMeasurement {
  targetPhysicalSessionProjectionId: string;
  denseRowsRemoved: number;
  lexicalSourceRowsRemoved: number;
  projectionRowsRemoved: number;
  replayWasIdempotent: boolean;
  completionCheck: 'reopen_and_verify';
  completionBasis: string;
  wholeGenerationRemoved: boolean;
}

interface PrototypeReport {
  generatedAt: string;
  scratchRoot: string;
  fixture: {
    sessions: number;
    entryAnchors: number;
    lexicalEvidenceOccurrences: number;
    denseEvidenceOccurrences: number;
    projectionRows: number;
    embeddingDimensions: number;
    productionMetadataReference: {
      physicalSessionFiles: number;
      sourceBytes: number;
      logicalChunks: number;
      sourceBytesPerChunk: number;
    };
  };
  lifecycle: {
    initialBuildMilliseconds: number;
    appendMilliseconds: number;
    optimizeMilliseconds: number;
    closeMilliseconds: number;
    reopenMilliseconds: number;
    replaySkippedRows: number;
    closeReturnType: string;
  };
  sizesAfterInitialBuild: StoreSizes;
  sizesAfterAppend: StoreSizes;
  sizesAfterOptimize: StoreSizes;
  peakBytesObservedDuringOptimize: number;
  validation: {
    logicalLexicalSourceRows: number;
    logicalDenseRows: number;
    logicalProjectionRows: number;
    ordinaryFtsCanary: boolean;
    identifierFtsCanary: boolean;
    exactOccurrenceCanary: boolean;
    denseCanary: boolean;
    crossStoreMembershipCanary: boolean;
    denseIndexCompleteness: number;
  };
  neighborhood: NeighborhoodMeasurement;
  replay: ReplayMeasurement[];
  recoveryFaults: RecoveryFaultMeasurement[];
  deletion: DeletionMeasurement;
  durabilityBoundary: {
    nativeDurabilityEstablished: false;
    lightweightPolicy: string;
    optionalNativeCapability: string;
    deferredFaults: string[];
  };
  verdict: string;
}

interface OpenGenerationStores {
  lexicalSource: ZVecCollection;
  dense: ZVecCollection;
  projections: ZVecCollection;
}

interface PrototypeOperationState {
  phase:
    | 'building'
    | 'built'
    | 'appended'
    | 'validated'
    | 'deleting_dense'
    | 'deleting_lexical_source'
    | 'deleting_projections'
    | 'verification_complete'
    | 'generation_removed';
  targetPhysicalSessionProjectionId: string | null;
}

const productionMetadataReference = Object.freeze({
  physicalSessionFiles: 3_500,
  sourceBytes: 2_346_218_035,
  logicalChunks: 1_175_836,
  sourceBytesPerChunk: 1_995.36,
});

const lexicalSourceFields = [
  {
    name: 'rowKind',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'checksum', dataType: ZVecDataType.STRING },
  {
    name: 'physicalSessionProjectionId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  {
    name: 'logicalSessionId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  {
    name: 'entryId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'parentEntryId', dataType: ZVecDataType.STRING },
  { name: 'branchPathLeafIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'occurrenceIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'evidenceKind', dataType: ZVecDataType.STRING },
  { name: 'sourceLineStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceLineEnd', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockEnd', dataType: ZVecDataType.INT32 },
  {
    name: 'content',
    dataType: ZVecDataType.STRING,
    indexParams: {
      indexType: ZVecIndexType.FTS,
      tokenizerName: 'standard',
      filters: ['lowercase'],
    },
  },
  {
    name: 'identifierContent',
    dataType: ZVecDataType.STRING,
    indexParams: {
      indexType: ZVecIndexType.FTS,
      tokenizerName: 'standard',
      filters: [],
    },
  },
];

const projectionFields = [
  {
    name: 'projectionKind',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  {
    name: 'physicalSessionProjectionId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'logicalSessionId', dataType: ZVecDataType.STRING },
  { name: 'revision', dataType: ZVecDataType.INT32 },
  { name: 'projectionJson', dataType: ZVecDataType.STRING },
];

function createDigest(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function createEmbedding(seed: number, dimensions: number): number[] {
  let state = Math.imul(seed + 1, 0x9e3779b1) >>> 0;
  const vector = Array.from({ length: dimensions }, () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x7fffffff - 1;
  });
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return vector.map((value) => value / magnitude);
}

function createRepresentativeContent(seed: number): string {
  const words = [
    'predictable',
    'recall',
    'evidence',
    'source',
    'neighborhood',
    'generation',
    'projection',
    'lexical',
    'dense',
    'checkpoint',
    'branch',
    'provenance',
  ];
  const wordCount = 96 + (seed % 320);
  return Array.from({ length: wordCount }, (_, index) => words[(seed + index) % words.length]).join(
    ' ',
  );
}

/** Builds deterministic schema-representative evidence with a main and abandoned branch per session. */
function createRepresentativeFixture(
  sessionCount: number,
  dimensions: number,
  sessionOffset = 0,
): RepresentativeFixture {
  const entryAnchors: EntryAnchorFixture[] = [];
  const evidenceOccurrences: EvidenceOccurrenceFixture[] = [];
  const projections: ProjectionFixture[] = [];
  let firstMainEndpoint = '';
  let firstAbandonedEndpoint = '';
  let firstSharedOccurrence = '';
  let firstMainOnlyOccurrence = '';

  for (let localSessionIndex = 0; localSessionIndex < sessionCount; localSessionIndex += 1) {
    const sessionIndex = sessionOffset + localSessionIndex;
    const physicalSessionProjectionId = `physical-${sessionIndex}`;
    const logicalSessionId = `logical-${sessionIndex}`;
    const mainEntries = Array.from({ length: 18 }, (_, index) => `s${sessionIndex}-main-${index}`);
    const branchEntries = Array.from(
      { length: 6 },
      (_, index) => `s${sessionIndex}-branch-${index}`,
    );
    const mainEndpoint = mainEntries.at(-1) ?? '';
    const abandonedEndpoint = branchEntries.at(-1) ?? '';
    firstMainEndpoint ||= mainEndpoint;
    firstAbandonedEndpoint ||= abandonedEndpoint;

    const entries = [
      ...mainEntries.map((entryId, index) => ({
        entryId,
        parentEntryId: index === 0 ? '' : (mainEntries[index - 1] ?? ''),
        branchPathLeafIds: index <= 7 ? [mainEndpoint, abandonedEndpoint] : [mainEndpoint],
      })),
      ...branchEntries.map((entryId, index) => ({
        entryId,
        parentEntryId: index === 0 ? (mainEntries[7] ?? '') : (branchEntries[index - 1] ?? ''),
        branchPathLeafIds: [abandonedEndpoint],
      })),
    ];

    for (const [entryIndex, entry] of entries.entries()) {
      const conversationOccurrenceId = `occurrence-${entry.entryId}-conversation`;
      const occurrenceIds = [conversationOccurrenceId];
      const content = createRepresentativeContent(sessionIndex * 100 + entryIndex);
      const denseSearchable = entryIndex % 5 !== 0;
      evidenceOccurrences.push({
        occurrenceId: conversationOccurrenceId,
        checksum: createDigest(content),
        physicalSessionProjectionId,
        logicalSessionId,
        entryId: entry.entryId,
        evidenceKind: 'conversation',
        content,
        identifierContent: `RecallTopology_${sessionIndex}_${entryIndex} src/session-${sessionIndex}.ts`,
        sourceLineStart: entryIndex + 2,
        sourceLineEnd: entryIndex + 2,
        sourceBlockStart: 0,
        sourceBlockEnd: 0,
        embedding: denseSearchable
          ? createEmbedding(sessionIndex * 100 + entryIndex, dimensions)
          : null,
      });
      if (entryIndex % 7 === 0) {
        const toolOccurrenceId = `occurrence-${entry.entryId}-tool`;
        const toolContent = JSON.stringify({
          command: `rg RecallTopology_${sessionIndex}_${entryIndex} src`,
          result: `src/session-${sessionIndex}.ts:${entryIndex + 2}`,
        });
        occurrenceIds.push(toolOccurrenceId);
        evidenceOccurrences.push({
          occurrenceId: toolOccurrenceId,
          checksum: createDigest(toolContent),
          physicalSessionProjectionId,
          logicalSessionId,
          entryId: entry.entryId,
          evidenceKind: 'tool_result',
          content: toolContent,
          identifierContent: `RecallTopology_${sessionIndex}_${entryIndex} src/session-${sessionIndex}.ts`,
          sourceLineStart: entryIndex + 2,
          sourceLineEnd: entryIndex + 2,
          sourceBlockStart: 1,
          sourceBlockEnd: 1,
          embedding: null,
        });
      }
      entryAnchors.push({
        anchorId: createEntryAnchorId(logicalSessionId, entry.entryId),
        physicalSessionProjectionId,
        logicalSessionId,
        entryId: entry.entryId,
        parentEntryId: entry.parentEntryId,
        branchPathLeafIds: entry.branchPathLeafIds,
        occurrenceIds,
        sourceLineStart: entryIndex + 2,
        sourceLineEnd: entryIndex + 2,
      });
      if (sessionIndex === 0 && entry.entryId === mainEntries[7]) {
        firstSharedOccurrence = conversationOccurrenceId;
      }
      if (sessionIndex === 0 && entry.entryId === mainEntries[12]) {
        firstMainOnlyOccurrence = conversationOccurrenceId;
      }
    }

    const physicalProjection = {
      projectionId: physicalSessionProjectionId,
      projectionKind: 'physical' as const,
      physicalSessionProjectionId,
      logicalSessionId: '',
      revision: 1,
      projectionJson: JSON.stringify({
        appendCursorBytes: 150_000 + sessionIndex * 10_000,
        logicalSessionIds: [logicalSessionId],
        markerCheckpoint: sessionIndex,
      }),
    };
    const logicalProjection = {
      projectionId: `projection-${logicalSessionId}`,
      projectionKind: 'logical' as const,
      physicalSessionProjectionId,
      logicalSessionId,
      revision: 1,
      projectionJson: JSON.stringify({
        effectiveLeafEntryId: mainEndpoint,
        branchEndpointEntryIds: [mainEndpoint, abandonedEndpoint],
        eligibleEntryCount: entries.length,
      }),
    };
    projections.push(physicalProjection, logicalProjection);
  }

  return {
    entryAnchors,
    evidenceOccurrences,
    projections,
    mainBranchEndpointEntryId: firstMainEndpoint,
    abandonedBranchEndpointEntryId: firstAbandonedEndpoint,
    sharedAncestorOccurrenceId: firstSharedOccurrence,
    mainOnlyOccurrenceId: firstMainOnlyOccurrence,
  };
}

function createEntryAnchorId(logicalSessionId: string, entryId: string): string {
  return `entry-anchor-${createDigest(`${logicalSessionId}\0${entryId}`).slice(0, 32)}`;
}

function serializeEntryAnchor(anchor: EntryAnchorFixture): ZVecDocInput {
  return {
    id: anchor.anchorId,
    fields: {
      rowKind: 'entry_anchor',
      checksum: createDigest(JSON.stringify(anchor)),
      physicalSessionProjectionId: anchor.physicalSessionProjectionId,
      logicalSessionId: anchor.logicalSessionId,
      entryId: anchor.entryId,
      parentEntryId: anchor.parentEntryId,
      branchPathLeafIds: anchor.branchPathLeafIds,
      occurrenceIds: anchor.occurrenceIds,
      evidenceKind: '',
      sourceLineStart: anchor.sourceLineStart,
      sourceLineEnd: anchor.sourceLineEnd,
      sourceBlockStart: -1,
      sourceBlockEnd: -1,
      content: '',
      identifierContent: '',
    },
  };
}

function serializeLexicalEvidence(occurrence: EvidenceOccurrenceFixture): ZVecDocInput {
  return {
    id: occurrence.occurrenceId,
    fields: {
      rowKind: 'evidence',
      checksum: occurrence.checksum,
      physicalSessionProjectionId: occurrence.physicalSessionProjectionId,
      logicalSessionId: occurrence.logicalSessionId,
      entryId: occurrence.entryId,
      parentEntryId: '',
      branchPathLeafIds: [],
      occurrenceIds: [],
      evidenceKind: occurrence.evidenceKind,
      sourceLineStart: occurrence.sourceLineStart,
      sourceLineEnd: occurrence.sourceLineEnd,
      sourceBlockStart: occurrence.sourceBlockStart,
      sourceBlockEnd: occurrence.sourceBlockEnd,
      content: occurrence.content,
      identifierContent: occurrence.identifierContent,
    },
  };
}

function serializeDenseEvidence(occurrence: EvidenceOccurrenceFixture): ZVecDocInput {
  if (occurrence.embedding === null) {
    throw new Error(
      `Prototype dense serialization rejected lexical-only ${occurrence.occurrenceId}`,
    );
  }
  return {
    id: occurrence.occurrenceId,
    vectors: { embedding: occurrence.embedding },
    fields: {
      checksum: occurrence.checksum,
      logicalSessionId: occurrence.logicalSessionId,
      entryId: occurrence.entryId,
    },
  };
}

function serializeProjection(projection: ProjectionFixture): ZVecDocInput {
  return {
    id: projection.projectionId,
    fields: {
      projectionKind: projection.projectionKind,
      physicalSessionProjectionId: projection.physicalSessionProjectionId,
      logicalSessionId: projection.logicalSessionId,
      revision: projection.revision,
      projectionJson: projection.projectionJson,
    },
  };
}

function createGenerationStores(dimensions: number): OpenGenerationStores {
  mkdirSync(GENERATION_ROOT, { recursive: true });
  const lexicalSource = ZVecCreateAndOpen(
    join(GENERATION_ROOT, 'lexical-source'),
    new ZVecCollectionSchema({ name: 'recall_lexical_source', fields: lexicalSourceFields }),
  );
  const dense = ZVecCreateAndOpen(
    join(GENERATION_ROOT, 'dense'),
    new ZVecCollectionSchema({
      name: 'recall_dense',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: dimensions,
        indexParams: {
          indexType: ZVecIndexType.HNSW,
          metricType: ZVecMetricType.COSINE,
          m: 16,
          efConstruction: 100,
        },
      },
      fields: [
        { name: 'checksum', dataType: ZVecDataType.STRING },
        { name: 'logicalSessionId', dataType: ZVecDataType.STRING },
        { name: 'entryId', dataType: ZVecDataType.STRING },
      ],
    }),
  );
  const projections = ZVecCreateAndOpen(
    join(GENERATION_ROOT, 'projections'),
    new ZVecCollectionSchema({ name: 'recall_projections', fields: projectionFields }),
  );
  return { lexicalSource, dense, projections };
}

function openGenerationStores(): OpenGenerationStores {
  return {
    lexicalSource: ZVecOpen(join(GENERATION_ROOT, 'lexical-source')),
    dense: ZVecOpen(join(GENERATION_ROOT, 'dense')),
    projections: ZVecOpen(join(GENERATION_ROOT, 'projections')),
  };
}

function assertStatuses(operation: string, statuses: ZVecStatus | ZVecStatus[]): void {
  const statusList = Array.isArray(statuses) ? statuses : [statuses];
  for (const [index, status] of statusList.entries()) {
    if (!status.ok) {
      throw new Error(
        `Prototype ${operation} failed at position ${index} [${status.code}]: ${status.message}`,
      );
    }
  }
}

function insertBatches(collection: ZVecCollection, documents: ZVecDocInput[]): number {
  let successfulWrites = 0;
  for (let offset = 0; offset < documents.length; offset += WRITE_BATCH_SIZE) {
    const batch = documents.slice(offset, offset + WRITE_BATCH_SIZE);
    const statuses = collection.insertSync(batch);
    assertStatuses('insert', statuses);
    successfulWrites += statuses.length;
  }
  return successfulWrites;
}

function upsertBatches(collection: ZVecCollection, documents: ZVecDocInput[]): number {
  let successfulWrites = 0;
  for (let offset = 0; offset < documents.length; offset += WRITE_BATCH_SIZE) {
    const batch = documents.slice(offset, offset + WRITE_BATCH_SIZE);
    const statuses = collection.upsertSync(batch);
    assertStatuses('upsert', statuses);
    successfulWrites += statuses.length;
  }
  return successfulWrites;
}

function fetchVerifyInsert(
  collection: ZVecCollection,
  documents: ZVecDocInput[],
): {
  inserted: number;
  skipped: number;
} {
  let inserted = 0;
  let skipped = 0;
  for (let offset = 0; offset < documents.length; offset += WRITE_BATCH_SIZE) {
    const batch = documents.slice(offset, offset + WRITE_BATCH_SIZE);
    const fetched = collection.fetchSync({
      ids: batch.map(({ id }) => id),
      outputFields: ['checksum'],
      includeVector: false,
    });
    const missing: ZVecDocInput[] = [];
    for (const document of batch) {
      const existing = fetched[document.id];
      if (existing === undefined) {
        missing.push(document);
        continue;
      }
      const expectedChecksum: unknown = document.fields?.checksum;
      if (expectedChecksum !== undefined && existing.fields.checksum !== expectedChecksum) {
        throw new Error(`Prototype immutable identity mismatch for ${document.id}`);
      }
      skipped += 1;
    }
    if (missing.length > 0) {
      inserted += insertBatches(collection, missing);
    }
  }
  return { inserted, skipped };
}

function closeStores(stores: OpenGenerationStores): { milliseconds: number; returnType: string } {
  const startedAt = performance.now();
  const returns = [
    stores.projections.closeSync(),
    stores.dense.closeSync(),
    stores.lexicalSource.closeSync(),
  ];
  return {
    milliseconds: performance.now() - startedAt,
    returnType: returns.map((value) => typeof value).join(','),
  };
}

function isMissingFileError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && Reflect.get(error, 'code') === 'ENOENT';
}

function measureComponentSize(path: string): ComponentSize {
  if (!existsSync(path)) {
    return { files: 0, apparentBytes: 0, allocatedBytes: 0 };
  }
  let files = 0;
  let apparentBytes = 0;
  let allocatedBytes = 0;
  const visit = (currentPath: string): void => {
    let stat;
    try {
      stat = statSync(currentPath);
    } catch (error) {
      if (isMissingFileError(error)) {
        return;
      }
      throw error;
    }
    if (stat.isDirectory()) {
      let names: string[];
      try {
        names = readdirSync(currentPath);
      } catch (error) {
        if (isMissingFileError(error)) {
          return;
        }
        throw error;
      }
      for (const name of names) {
        visit(join(currentPath, name));
      }
      return;
    }
    files += 1;
    apparentBytes += stat.size;
    allocatedBytes += stat.blocks * 512;
  };
  visit(path);
  return { files, apparentBytes, allocatedBytes };
}

function measureStoreSizes(): StoreSizes {
  return {
    lexicalSource: measureComponentSize(join(GENERATION_ROOT, 'lexical-source')),
    dense: measureComponentSize(join(GENERATION_ROOT, 'dense')),
    projections: measureComponentSize(join(GENERATION_ROOT, 'projections')),
    generation: measureComponentSize(GENERATION_ROOT),
  };
}

function writeAtomicJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp`;
  const fileDescriptor = openSync(temporaryPath, 'w');
  try {
    writeFileSync(fileDescriptor, `${JSON.stringify(value, null, 2)}\n`);
    fsyncSync(fileDescriptor);
  } finally {
    closeSync(fileDescriptor);
  }
  renameSync(temporaryPath, path);
  const directoryDescriptor = openSync(dirname(path), 'r');
  try {
    fsyncSync(directoryDescriptor);
  } finally {
    closeSync(directoryDescriptor);
  }
}

function writeOperationState(state: PrototypeOperationState): void {
  writeAtomicJson(join(GENERATION_ROOT, 'operation-state.json'), state);
}

function writePrototypeManifest(dimensions: number): void {
  writeAtomicJson(join(GENERATION_ROOT, 'index-manifest.json'), {
    prototype: true,
    generationId: GENERATION_ID,
    topologyVersion: 1,
    embeddingDimensions: dimensions,
    stores: ['lexical-source', 'dense', 'projections'],
  });
}

async function optimizeStores(stores: OpenGenerationStores): Promise<{
  milliseconds: number;
  peakGenerationBytes: number;
}> {
  let peakGenerationBytes = measureStoreSizes().generation.apparentBytes;
  const sampler = setInterval(() => {
    peakGenerationBytes = Math.max(
      peakGenerationBytes,
      measureStoreSizes().generation.apparentBytes,
    );
  }, 10);
  const startedAt = performance.now();
  try {
    await stores.lexicalSource.optimize();
    await stores.dense.optimize();
    await stores.projections.optimize();
  } finally {
    clearInterval(sampler);
  }
  peakGenerationBytes = Math.max(peakGenerationBytes, measureStoreSizes().generation.apparentBytes);
  return { milliseconds: performance.now() - startedAt, peakGenerationBytes };
}

function readStringField(document: ZVecDoc, name: string): string {
  const value: unknown = document.fields[name];
  if (typeof value !== 'string') {
    throw new Error(`Prototype field ${name} was not a string for ${document.id}`);
  }
  return value;
}

function readStringArrayField(document: ZVecDoc, name: string): string[] {
  const value: unknown = document.fields[name];
  if (!Array.isArray(value)) {
    throw new Error(`Prototype field ${name} was not a string array for ${document.id}`);
  }
  const strings: string[] = [];
  for (const itemValue of value) {
    const item: unknown = itemValue;
    if (typeof item !== 'string') {
      throw new Error(`Prototype field ${name} was not a string array for ${document.id}`);
    }
    strings.push(item);
  }
  return strings;
}

function fetchRequiredDocument(collection: ZVecCollection, id: string): ZVecDoc {
  const document = collection.fetchSync({ ids: id, includeVector: false })[id];
  if (document === undefined) {
    throw new Error(`Prototype required document missing: ${id}`);
  }
  return document;
}

/** Expands one exact occurrence on the selected branch path through immutable entry anchors. */
function expandIndexedSourceNeighborhood(
  lexicalSource: ZVecCollection,
  occurrenceId: string,
  selectedEndpointEntryId: string,
  before: number,
  after: number,
): NeighborhoodMeasurement {
  const occurrence = fetchRequiredDocument(lexicalSource, occurrenceId);
  const logicalSessionId = readStringField(occurrence, 'logicalSessionId');
  const anchorEntryId = readStringField(occurrence, 'entryId');
  const reversePath: ZVecDoc[] = [];
  let nextEntryId = selectedEndpointEntryId;
  const visited = new Set<string>();
  while (nextEntryId !== '') {
    if (visited.has(nextEntryId)) {
      throw new Error(`Prototype source neighborhood cycle at ${nextEntryId}`);
    }
    visited.add(nextEntryId);
    const entryAnchor = fetchRequiredDocument(
      lexicalSource,
      createEntryAnchorId(logicalSessionId, nextEntryId),
    );
    if (readStringField(entryAnchor, 'logicalSessionId') !== logicalSessionId) {
      throw new Error('Prototype source neighborhood crossed a logical session');
    }
    reversePath.push(entryAnchor);
    nextEntryId = readStringField(entryAnchor, 'parentEntryId');
  }
  const selectedPath = reversePath.toReversed();
  const anchorIndex = selectedPath.findIndex(
    (entryAnchor) => readStringField(entryAnchor, 'entryId') === anchorEntryId,
  );
  if (anchorIndex < 0) {
    throw new Error(
      `Prototype occurrence ${occurrenceId} is not on selected endpoint ${selectedEndpointEntryId}`,
    );
  }
  const neighborhoodAnchors = selectedPath.slice(
    Math.max(0, anchorIndex - before),
    anchorIndex + after + 1,
  );
  const occurrenceIds = neighborhoodAnchors.flatMap((entryAnchor) =>
    readStringArrayField(entryAnchor, 'occurrenceIds'),
  );
  const neighborhoodOccurrences = lexicalSource.fetchSync({
    ids: occurrenceIds,
    outputFields: ['logicalSessionId'],
    includeVector: false,
  });
  return {
    selectedEndpointEntryId,
    entryIds: neighborhoodAnchors.map((entryAnchor) => readStringField(entryAnchor, 'entryId')),
    occurrenceIds,
    stayedInLogicalSession: Object.values(neighborhoodOccurrences).every(
      (document) => readStringField(document, 'logicalSessionId') === logicalSessionId,
    ),
    unrelatedBranchRejected: false,
  };
}

function validateGeneration(
  stores: OpenGenerationStores,
  fixture: RepresentativeFixture,
): PrototypeReport['validation'] {
  const expectedLexicalSourceRows =
    fixture.entryAnchors.length + fixture.evidenceOccurrences.length;
  const denseOccurrences = fixture.evidenceOccurrences.filter(
    ({ embedding }) => embedding !== null,
  );
  const ordinaryFts = stores.lexicalSource.querySync({
    fieldName: 'content',
    fts: { matchString: 'predictable recall evidence' },
    filter: "rowKind = 'evidence'",
    topk: 5,
    outputFields: ['rowKind'],
    includeVector: false,
    params: { indexType: ZVecIndexType.FTS, defaultOperator: 'AND' },
  });
  const identifierFts = stores.lexicalSource.querySync({
    fieldName: 'identifierContent',
    fts: { matchString: 'RecallTopology_0_0' },
    filter: "rowKind = 'evidence'",
    topk: 5,
    outputFields: ['identifierContent'],
    includeVector: false,
    params: { indexType: ZVecIndexType.FTS, defaultOperator: 'AND' },
  });
  const wrongCaseIdentifierFts = stores.lexicalSource.querySync({
    fieldName: 'identifierContent',
    fts: { matchString: 'recalltopology_0_0' },
    filter: "rowKind = 'evidence'",
    topk: 5,
    outputFields: ['identifierContent'],
    includeVector: false,
    params: { indexType: ZVecIndexType.FTS, defaultOperator: 'AND' },
  });
  const exactOccurrence = fetchRequiredDocument(
    stores.lexicalSource,
    fixture.sharedAncestorOccurrenceId,
  );
  const denseCanaryFixture = denseOccurrences[0];
  if (denseCanaryFixture === undefined || denseCanaryFixture.embedding === null) {
    throw new Error('Prototype dense canary fixture missing');
  }
  const denseSearch = stores.dense.querySync({
    fieldName: 'embedding',
    vector: denseCanaryFixture.embedding,
    topk: 1,
    outputFields: ['checksum'],
    includeVector: false,
    params: { indexType: ZVecIndexType.HNSW, ef: 64 },
  });
  const denseIds = denseOccurrences
    .slice(0, WRITE_BATCH_SIZE)
    .map(({ occurrenceId }) => occurrenceId);
  const lexicalMembership = stores.lexicalSource.fetchSync({
    ids: denseIds,
    outputFields: ['checksum'],
    includeVector: false,
  });
  const denseMembership = stores.dense.fetchSync({
    ids: denseIds,
    outputFields: ['checksum'],
    includeVector: false,
  });
  const crossStoreMembershipCanary = denseIds.every(
    (id) =>
      lexicalMembership[id] !== undefined &&
      denseMembership[id] !== undefined &&
      lexicalMembership[id]?.fields.checksum === denseMembership[id]?.fields.checksum,
  );
  return {
    logicalLexicalSourceRows: stores.lexicalSource.stats.docCount,
    logicalDenseRows: stores.dense.stats.docCount,
    logicalProjectionRows: stores.projections.stats.docCount,
    ordinaryFtsCanary:
      ordinaryFts.length > 0 &&
      ordinaryFts.every((document) => document.fields.rowKind === 'evidence'),
    identifierFtsCanary: identifierFts.length > 0 && wrongCaseIdentifierFts.length === 0,
    exactOccurrenceCanary:
      readStringField(exactOccurrence, 'rowKind') === 'evidence' &&
      stores.lexicalSource.stats.docCount === expectedLexicalSourceRows,
    denseCanary: denseSearch[0]?.id === denseCanaryFixture.occurrenceId,
    crossStoreMembershipCanary,
    denseIndexCompleteness: stores.dense.stats.indexCompleteness.embedding ?? 0,
  };
}

function createReplayCollection(path: string): ZVecCollection {
  return ZVecCreateAndOpen(
    path,
    new ZVecCollectionSchema({
      name: 'replay_probe',
      fields: [
        { name: 'checksum', dataType: ZVecDataType.STRING },
        { name: 'content', dataType: ZVecDataType.STRING },
      ],
    }),
  );
}

function runReplayMeasurement(strategy: ReplayMeasurement['strategy']): ReplayMeasurement {
  const root = join(PROTOTYPE_ROOT, `replay-${strategy}`);
  rmSync(root, { recursive: true, force: true });
  const documents = Array.from({ length: 256 }, (_, index) => {
    const content = createRepresentativeContent(index);
    return {
      id: `replay-${index}`,
      fields: { checksum: createDigest(content), content },
    } satisfies ZVecDocInput;
  });
  let collection = createReplayCollection(root);
  let successfulPhysicalWriteVersions = insertBatches(collection, documents);
  collection.closeSync();
  const sizesAfterPasses = [measureComponentSize(root).apparentBytes];
  for (let pass = 0; pass < 3; pass += 1) {
    collection = ZVecOpen(root);
    if (strategy === 'blind_upsert') {
      successfulPhysicalWriteVersions += upsertBatches(collection, documents);
    } else {
      successfulPhysicalWriteVersions += fetchVerifyInsert(collection, documents).inserted;
    }
    collection.closeSync();
    sizesAfterPasses.push(measureComponentSize(root).apparentBytes);
  }
  collection = ZVecOpen(root);
  collection.optimizeSync();
  collection.closeSync();
  const sizeAfterOptimize = measureComponentSize(root).apparentBytes;
  return {
    strategy,
    logicalRows: documents.length,
    successfulPhysicalWriteVersions,
    sizesAfterPasses,
    sizeAfterOptimize,
  };
}

async function spawnCrashWriter(collectionPath: string): Promise<void> {
  const scriptPath = fileURLToPath(import.meta.url);
  const child = spawn(
    process.execPath,
    ['--import', 'tsx', scriptPath, '--child-crash-writer', collectionPath],
    { stdio: ['ignore', 'pipe', 'pipe'] },
  );
  let output = '';
  let errorOutput = '';
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (chunk: string) => {
    output += chunk;
  });
  child.stderr.on('data', (chunk: string) => {
    errorOutput += chunk;
  });
  await new Promise<void>((resolveReady, rejectReady) => {
    const deadline = setTimeout(() => {
      rejectReady(new Error(`Prototype crash writer timed out: ${errorOutput}`));
    }, 10_000);
    const checkReady = (): void => {
      if (output.includes('READY')) {
        clearTimeout(deadline);
        resolveReady();
      } else if (child.exitCode !== null) {
        clearTimeout(deadline);
        rejectReady(new Error(`Prototype crash writer exited early: ${errorOutput}`));
      } else {
        setTimeout(checkReady, 10);
      }
    };
    checkReady();
  });
  child.kill('SIGKILL');
  await new Promise<void>((resolveExit) => {
    child.once('exit', () => resolveExit());
  });
}

function findWalFile(root: string): string {
  const visit = (path: string): string | null => {
    for (const name of readdirSync(path)) {
      const childPath = join(path, name);
      const stat = statSync(childPath);
      if (stat.isDirectory()) {
        const nested = visit(childPath);
        if (nested !== null) {
          return nested;
        }
      } else if (name.endsWith('.wal')) {
        return childPath;
      }
    }
    return null;
  };
  const walPath = visit(root);
  if (walPath === null) {
    throw new Error(`Prototype WAL file missing beneath ${root}`);
  }
  return walPath;
}

function flipWalByte(path: string): void {
  const bytes = Buffer.from(readFileSync(path));
  const index = Math.floor(bytes.length / 2);
  const originalByte = bytes[index];
  if (originalByte === undefined) {
    throw new Error(`Prototype WAL file was empty: ${path}`);
  }
  bytes[index] = originalByte ^ 0xff;
  writeFileSync(path, bytes);
}

function recoverFaultVariant(
  variant: RecoveryFaultMeasurement['variant'],
  collectionPath: string,
): RecoveryFaultMeasurement {
  try {
    const collection = ZVecOpen(collectionPath);
    const logicalRows = collection.stats.docCount;
    const fetchedRows = Object.keys(
      collection.fetchSync(Array.from({ length: CRASH_ROW_COUNT }, (_, index) => `crash-${index}`)),
    ).length;
    collection.closeSync();
    return { variant, opened: true, logicalRows, fetchedRows, error: null };
  } catch (error) {
    return {
      variant,
      opened: false,
      logicalRows: null,
      fetchedRows: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function runRecoveryFaultMeasurements(): Promise<RecoveryFaultMeasurement[]> {
  const baseRoot = join(PROTOTYPE_ROOT, 'recovery-fault-base');
  rmSync(baseRoot, { recursive: true, force: true });
  mkdirSync(baseRoot, { recursive: true });
  const baseCollectionPath = join(baseRoot, 'collection');
  await spawnCrashWriter(baseCollectionPath);

  const variants: RecoveryFaultMeasurement['variant'][] = [
    'intact',
    'truncated',
    'flipped',
    'missing',
    'unreadable',
  ];
  for (const variant of variants) {
    cpSync(baseRoot, join(PROTOTYPE_ROOT, `recovery-${variant}`), { recursive: true });
  }
  const truncatedWal = findWalFile(join(PROTOTYPE_ROOT, 'recovery-truncated', 'collection'));
  truncateSync(truncatedWal, Math.max(0, statSync(truncatedWal).size - 17));
  flipWalByte(findWalFile(join(PROTOTYPE_ROOT, 'recovery-flipped', 'collection')));
  rmSync(findWalFile(join(PROTOTYPE_ROOT, 'recovery-missing', 'collection')));
  const unreadableWal = findWalFile(join(PROTOTYPE_ROOT, 'recovery-unreadable', 'collection'));
  chmodSync(unreadableWal, 0o000);

  const measurements = variants.map((variant) =>
    recoverFaultVariant(variant, join(PROTOTYPE_ROOT, `recovery-${variant}`, 'collection')),
  );
  chmodSync(unreadableWal, 0o600);
  return measurements;
}

async function runCrashWriterChild(collectionPath: string): Promise<never> {
  const collection = ZVecCreateAndOpen(
    collectionPath,
    new ZVecCollectionSchema({
      name: 'crash_recovery_probe',
      fields: [{ name: 'content', dataType: ZVecDataType.STRING }],
    }),
  );
  const documents = Array.from({ length: CRASH_ROW_COUNT }, (_, index) => ({
    id: `crash-${index}`,
    fields: { content: `crash recovery evidence ${index} ${'x'.repeat(512)}` },
  }));
  assertStatuses('crash writer insert', collection.insertSync(documents));
  process.stdout.write('READY\n');
  await new Promise(() => {
    setInterval(() => undefined, 1_000);
  });
  throw new Error('unreachable');
}

function deleteIds(collection: ZVecCollection, ids: string[]): number {
  let removed = 0;
  for (let offset = 0; offset < ids.length; offset += WRITE_BATCH_SIZE) {
    const batch = ids.slice(offset, offset + WRITE_BATCH_SIZE);
    const existing = collection.fetchSync({ ids: batch, outputFields: [], includeVector: false });
    const existingIds = Object.keys(existing);
    if (existingIds.length === 0) {
      continue;
    }
    assertStatuses('delete', collection.deleteSync(existingIds));
    removed += existingIds.length;
  }
  return removed;
}

function runResumableDeletion(
  fixture: RepresentativeFixture,
  dimensions: number,
): DeletionMeasurement {
  const targetPhysicalSessionProjectionId = 'physical-0';
  const targetEvidence = fixture.evidenceOccurrences.filter(
    ({ physicalSessionProjectionId }) =>
      physicalSessionProjectionId === targetPhysicalSessionProjectionId,
  );
  const targetAnchors = fixture.entryAnchors.filter(
    ({ physicalSessionProjectionId }) =>
      physicalSessionProjectionId === targetPhysicalSessionProjectionId,
  );
  const targetProjections = fixture.projections.filter(
    ({ physicalSessionProjectionId }) =>
      physicalSessionProjectionId === targetPhysicalSessionProjectionId,
  );
  const denseIds = targetEvidence
    .filter(({ embedding }) => embedding !== null)
    .map(({ occurrenceId }) => occurrenceId);
  const lexicalSourceIds = [
    ...targetEvidence.map(({ occurrenceId }) => occurrenceId),
    ...targetAnchors.map(({ anchorId }) => anchorId),
  ];
  const projectionIds = targetProjections.map(({ projectionId }) => projectionId);

  writeOperationState({ phase: 'deleting_dense', targetPhysicalSessionProjectionId });
  let stores = openGenerationStores();
  const denseRowsRemoved = deleteIds(stores.dense, denseIds);
  closeStores(stores);
  stores = openGenerationStores();
  const denseDeletionVerified = Object.keys(stores.dense.fetchSync(denseIds)).length === 0;
  closeStores(stores);
  if (!denseDeletionVerified) {
    throw new Error('Prototype dense deletion verification failed');
  }

  writeOperationState({
    phase: 'deleting_lexical_source',
    targetPhysicalSessionProjectionId,
  });
  stores = openGenerationStores();
  const lexicalSourceRowsRemoved = deleteIds(stores.lexicalSource, lexicalSourceIds);
  closeStores(stores);
  stores = openGenerationStores();
  const lexicalDeletionVerified =
    Object.keys(stores.lexicalSource.fetchSync(lexicalSourceIds)).length === 0;
  closeStores(stores);
  if (!lexicalDeletionVerified) {
    throw new Error('Prototype lexical/source deletion verification failed');
  }

  writeOperationState({ phase: 'deleting_projections', targetPhysicalSessionProjectionId });
  stores = openGenerationStores();
  const projectionRowsRemoved = deleteIds(stores.projections, projectionIds);
  closeStores(stores);
  stores = openGenerationStores();
  const projectionDeletionVerified =
    Object.keys(stores.projections.fetchSync(projectionIds)).length === 0;
  const replayRemovedRows =
    deleteIds(stores.dense, denseIds) +
    deleteIds(stores.lexicalSource, lexicalSourceIds) +
    deleteIds(stores.projections, projectionIds);
  closeStores(stores);
  if (!projectionDeletionVerified) {
    throw new Error('Prototype projection deletion verification failed');
  }

  const completionBasis =
    'Advance the checkpoint only after reopen verifies the expected IDs are absent. If broader validation fails, replay the source and reuse valid rows; rebuild the whole generation only when the damage cannot be isolated.';
  writeOperationState({
    phase: 'verification_complete',
    targetPhysicalSessionProjectionId,
  });
  writeAtomicJson(join(GENERATION_ROOT, 'deletion-verification.json'), {
    targetPhysicalSessionProjectionId,
    denseRowsRemoved,
    lexicalSourceRowsRemoved,
    projectionRowsRemoved,
    replayRemovedRows,
    completionCheck: 'reopen_and_verify',
    completionBasis,
    dimensions,
  });
  return {
    targetPhysicalSessionProjectionId,
    denseRowsRemoved,
    lexicalSourceRowsRemoved,
    projectionRowsRemoved,
    replayWasIdempotent: replayRemovedRows === 0,
    completionCheck: 'reopen_and_verify',
    completionBasis,
    wholeGenerationRemoved: false,
  };
}

function removeWholeGeneration(): boolean {
  const generationsRoot = dirname(GENERATION_ROOT);
  const tombstonePath = `${GENERATION_ROOT}.deleting`;
  rmSync(tombstonePath, { recursive: true, force: true });
  renameSync(GENERATION_ROOT, tombstonePath);
  const generationsDescriptor = openSync(generationsRoot, 'r');
  try {
    fsyncSync(generationsDescriptor);
  } finally {
    closeSync(generationsDescriptor);
  }
  rmSync(tombstonePath, { recursive: true, force: true });
  const finalDescriptor = openSync(generationsRoot, 'r');
  try {
    fsyncSync(finalDescriptor);
  } finally {
    closeSync(finalDescriptor);
  }
  return !existsSync(GENERATION_ROOT) && !existsSync(tombstonePath);
}

function formatSizesTable(label: string, sizes: StoreSizes): string {
  return [
    `### ${label}`,
    '',
    '| Component | Files | Apparent bytes | Allocated bytes |',
    '| --- | ---: | ---: | ---: |',
    `| lexical-source | ${sizes.lexicalSource.files} | ${byteFormatter.format(sizes.lexicalSource.apparentBytes)} | ${byteFormatter.format(sizes.lexicalSource.allocatedBytes)} |`,
    `| dense | ${sizes.dense.files} | ${byteFormatter.format(sizes.dense.apparentBytes)} | ${byteFormatter.format(sizes.dense.allocatedBytes)} |`,
    `| projections | ${sizes.projections.files} | ${byteFormatter.format(sizes.projections.apparentBytes)} | ${byteFormatter.format(sizes.projections.allocatedBytes)} |`,
    `| whole generation | ${sizes.generation.files} | ${byteFormatter.format(sizes.generation.apparentBytes)} | ${byteFormatter.format(sizes.generation.allocatedBytes)} |`,
  ].join('\n');
}

function formatPrototypeReport(report: PrototypeReport): string {
  const replayRows = report.replay
    .map(
      (measurement) =>
        `| ${measurement.strategy} | ${measurement.logicalRows} | ${measurement.successfulPhysicalWriteVersions} | ${measurement.sizesAfterPasses.map((bytes) => byteFormatter.format(bytes)).join(' → ')} | ${byteFormatter.format(measurement.sizeAfterOptimize)} |`,
    )
    .join('\n');
  const faultRows = report.recoveryFaults
    .map(
      (measurement) =>
        `| ${measurement.variant} | ${measurement.opened} | ${measurement.logicalRows ?? '—'} | ${measurement.fetchedRows ?? '—'} | ${measurement.error ?? 'none'} |`,
    )
    .join('\n');
  return `# PROTOTYPE measurements — predictable recall storage topology

Generated ${report.generatedAt} with \`@zvec/zvec\` 0.6.0 on disposable scratch data.

## Verdict

${report.verdict}

The smallest useful topology is three zvec collections plus application-owned generation metadata:

\`lexical-source/\` stores vector-free lexical evidence and immutable entry anchors; \`dense/\` stores only embedded occurrence IDs; \`projections/\` isolates mutable ingestion state. Stable occurrence IDs join lexical and dense evidence. Entry anchors provide direct source-neighborhood traversal without reopening JSONL.

This topology is acceptable for a rebuildable index. Activate only a reopened and validated generation, and keep the previous valid generation until activation succeeds. After an interrupted build, replay the source, reuse rows whose occurrence ID, embedding profile, and content checksum still match, and re-embed only missing or damaged rows. Rebuild the whole generation only when the damage cannot be isolated.

## Representative fixture

- Sessions: ${report.fixture.sessions}
- Immutable entry anchors: ${report.fixture.entryAnchors}
- Lexical evidence occurrences: ${report.fixture.lexicalEvidenceOccurrences}
- Dense evidence occurrences: ${report.fixture.denseEvidenceOccurrences}
- Mutable projection rows: ${report.fixture.projectionRows}
- Stored embedding dimensions: ${report.fixture.embeddingDimensions}
- Production metadata reference (filesystem/index-state only): ${byteFormatter.format(report.fixture.productionMetadataReference.physicalSessionFiles)} physical files, ${byteFormatter.format(report.fixture.productionMetadataReference.sourceBytes)} source bytes, ${byteFormatter.format(report.fixture.productionMetadataReference.logicalChunks)} logical chunks, ${report.fixture.productionMetadataReference.sourceBytesPerChunk.toFixed(2)} source bytes per chunk.

The fixture reproduces the essential scalar/FTS provenance shape, 512-token-scale text, lexical-only tool evidence, dense-eligible conversation evidence, physical/logical projections, and a main plus abandoned branch. It is schema-representative, not a capacity forecast for private corpus text.

## Lifecycle

- Initial build: ${report.lifecycle.initialBuildMilliseconds.toFixed(1)} ms
- Incremental append: ${report.lifecycle.appendMilliseconds.toFixed(1)} ms
- Optimize: ${report.lifecycle.optimizeMilliseconds.toFixed(1)} ms
- Close: ${report.lifecycle.closeMilliseconds.toFixed(1)} ms; return types: \`${report.lifecycle.closeReturnType}\`
- Reopen: ${report.lifecycle.reopenMilliseconds.toFixed(1)} ms
- Immutable replay rows skipped after checksum verification: ${report.lifecycle.replaySkippedRows}
- Peak whole-generation bytes observed during optimize: ${byteFormatter.format(report.peakBytesObservedDuringOptimize)}

${formatSizesTable('After initial build', report.sizesAfterInitialBuild)}

${formatSizesTable('After append', report.sizesAfterAppend)}

${formatSizesTable('After optimize', report.sizesAfterOptimize)}

## Validation

| Check | Result |
| --- | --- |
| lexical/source logical rows | ${report.validation.logicalLexicalSourceRows} |
| dense logical rows | ${report.validation.logicalDenseRows} |
| projection logical rows | ${report.validation.logicalProjectionRows} |
| ordinary FTS canary | ${report.validation.ordinaryFtsCanary} |
| case-preserving identifier FTS canary | ${report.validation.identifierFtsCanary} |
| exact occurrence fetch | ${report.validation.exactOccurrenceCanary} |
| dense nearest-neighbor canary | ${report.validation.denseCanary} |
| dense membership/checksum subset of lexical evidence | ${report.validation.crossStoreMembershipCanary} |
| dense index completeness | ${report.validation.denseIndexCompleteness} |

## Source neighborhood

Selected endpoint: \`${report.neighborhood.selectedEndpointEntryId}\`

Path entries: ${report.neighborhood.entryIds.map((id) => `\`${id}\``).join(' → ')}

The expansion fetched ${report.neighborhood.occurrenceIds.length} exact evidence occurrences, stayed in one logical session: ${report.neighborhood.stayedInLogicalSession}, and rejected an anchor from an unrelated branch: ${report.neighborhood.unrelatedBranchRejected}.

## Replay amplification

“Physical write versions” counts successful engine write operations because zvec exposes only live logical \`docCount\`, not stale physical row count.

| Strategy | Live logical rows | Successful physical write versions | Apparent bytes after initial + 3 replays | Bytes after optimize |
| --- | ---: | ---: | --- | ---: |
${replayRows}

Fetch-and-verify plus insert-if-absent keeps immutable replay source-driven. Blind upsert preserves logical IDs but writes another physical version on every replay.

## Crash and recovery faults

Each variant began as the same SIGKILL residue containing ${CRASH_ROW_COUNT} successful writes.

| WAL variant | Open returned success | Logical rows | Exact rows fetched | Thrown error |
| --- | --- | ---: | ---: | --- |
${faultRows}

The intact process-crash path recovered. Truncated, CRC-flipped, missing, and unreadable WAL variants can return a successfully opened collection with partial or zero rows. Therefore a successful zvec open cannot by itself certify a generation. On restart, compare stored rows with the generation manifest and source identities, reuse valid embeddings, and recompute only missing or mismatched rows.

## Resumable split-store deletion

- Target physical session projection: \`${report.deletion.targetPhysicalSessionProjectionId}\`
- Dense rows removed first: ${report.deletion.denseRowsRemoved}
- Lexical evidence and entry-anchor rows removed second: ${report.deletion.lexicalSourceRowsRemoved}
- Logical and physical projection rows removed last: ${report.deletion.projectionRowsRemoved}
- A complete replay removed nothing: ${report.deletion.replayWasIdempotent}
- Whole generation rename-and-delete completed: ${report.deletion.wholeGenerationRemoved}
- Completion check: **${report.deletion.completionCheck}** — ${report.deletion.completionBasis}

## Lightweight durability policy

Native durability established: **${report.durabilityBoundary.nativeDurabilityEstablished}**. The session JSONL files protect the data; durability work here protects the time spent embedding it.

Required application policy: ${report.durabilityBoundary.lightweightPolicy}

Optional native improvement: ${report.durabilityBoundary.optionalNativeCapability}

Deferred unless a concrete need justifies them: ${report.durabilityBoundary.deferredFaults.join('; ')}.

## Decision implication

Adopt the three-store topology with zvec as the incumbent. Preserve these crash probes. Add cheap checkpoints or native durability only when they clearly reduce restart verification or re-embedding time without materially increasing code complexity or index size. Do not build a second recovery system to protect data that already exists in the session JSONL files.
`;
}

/** Runs the complete disposable topology, sizing, replay, fault, neighborhood, and deletion probe. */
async function runPrototype(): Promise<PrototypeReport> {
  rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  mkdirSync(join(PROTOTYPE_ROOT, 'generations'), { recursive: true });
  const dimensions = DEFAULT_EMBEDDING_DIMENSIONS;
  const initialFixture = createRepresentativeFixture(DEFAULT_SESSION_COUNT, dimensions);
  const appendFixture = createRepresentativeFixture(4, dimensions, DEFAULT_SESSION_COUNT);
  const completeFixture: RepresentativeFixture = {
    entryAnchors: [...initialFixture.entryAnchors, ...appendFixture.entryAnchors],
    evidenceOccurrences: [
      ...initialFixture.evidenceOccurrences,
      ...appendFixture.evidenceOccurrences,
    ],
    projections: [...initialFixture.projections, ...appendFixture.projections],
    mainBranchEndpointEntryId: initialFixture.mainBranchEndpointEntryId,
    abandonedBranchEndpointEntryId: initialFixture.abandonedBranchEndpointEntryId,
    sharedAncestorOccurrenceId: initialFixture.sharedAncestorOccurrenceId,
    mainOnlyOccurrenceId: initialFixture.mainOnlyOccurrenceId,
  };

  writeOperationState({ phase: 'building', targetPhysicalSessionProjectionId: null });
  writePrototypeManifest(dimensions);
  let stores = createGenerationStores(dimensions);
  const initialBuildStartedAt = performance.now();
  insertBatches(stores.lexicalSource, [
    ...initialFixture.entryAnchors.map(serializeEntryAnchor),
    ...initialFixture.evidenceOccurrences.map(serializeLexicalEvidence),
  ]);
  insertBatches(
    stores.dense,
    initialFixture.evidenceOccurrences
      .filter(({ embedding }) => embedding !== null)
      .map(serializeDenseEvidence),
  );
  insertBatches(stores.projections, initialFixture.projections.map(serializeProjection));
  const initialBuildMilliseconds = performance.now() - initialBuildStartedAt;
  const initialClose = closeStores(stores);
  writeOperationState({ phase: 'built', targetPhysicalSessionProjectionId: null });
  const sizesAfterInitialBuild = measureStoreSizes();

  const reopenStartedAt = performance.now();
  stores = openGenerationStores();
  const reopenMilliseconds = performance.now() - reopenStartedAt;
  const appendStartedAt = performance.now();
  const replay = fetchVerifyInsert(stores.lexicalSource, [
    ...initialFixture.entryAnchors.slice(0, 128).map(serializeEntryAnchor),
    ...initialFixture.evidenceOccurrences.slice(0, 128).map(serializeLexicalEvidence),
  ]);
  insertBatches(stores.lexicalSource, [
    ...appendFixture.entryAnchors.map(serializeEntryAnchor),
    ...appendFixture.evidenceOccurrences.map(serializeLexicalEvidence),
  ]);
  insertBatches(
    stores.dense,
    appendFixture.evidenceOccurrences
      .filter(({ embedding }) => embedding !== null)
      .map(serializeDenseEvidence),
  );
  insertBatches(stores.projections, appendFixture.projections.map(serializeProjection));
  const revisedProjections = initialFixture.projections.map((projection) => ({
    ...projection,
    revision: 2,
    projectionJson: JSON.stringify({ previous: projection.projectionJson, appended: true }),
  }));
  upsertBatches(stores.projections, revisedProjections.map(serializeProjection));
  const appendMilliseconds = performance.now() - appendStartedAt;
  closeStores(stores);
  writeOperationState({ phase: 'appended', targetPhysicalSessionProjectionId: null });
  const sizesAfterAppend = measureStoreSizes();

  stores = openGenerationStores();
  const optimize = await optimizeStores(stores);
  const validation = validateGeneration(stores, completeFixture);
  let neighborhood = expandIndexedSourceNeighborhood(
    stores.lexicalSource,
    completeFixture.sharedAncestorOccurrenceId,
    completeFixture.mainBranchEndpointEntryId,
    2,
    2,
  );
  let unrelatedBranchRejected = false;
  try {
    expandIndexedSourceNeighborhood(
      stores.lexicalSource,
      completeFixture.mainOnlyOccurrenceId,
      completeFixture.abandonedBranchEndpointEntryId,
      1,
      1,
    );
  } catch {
    unrelatedBranchRejected = true;
  }
  neighborhood = { ...neighborhood, unrelatedBranchRejected };
  const optimizedClose = closeStores(stores);
  writeOperationState({ phase: 'validated', targetPhysicalSessionProjectionId: null });
  const sizesAfterOptimize = measureStoreSizes();

  const replayMeasurements = [
    runReplayMeasurement('blind_upsert'),
    runReplayMeasurement('fetch_verify_insert'),
  ];
  const recoveryFaults = await runRecoveryFaultMeasurements();
  const deletion = runResumableDeletion(completeFixture, dimensions);
  deletion.wholeGenerationRemoved = removeWholeGeneration();

  const report: PrototypeReport = {
    generatedAt: new Date().toISOString(),
    scratchRoot: PROTOTYPE_ROOT,
    fixture: {
      sessions: DEFAULT_SESSION_COUNT + 4,
      entryAnchors: completeFixture.entryAnchors.length,
      lexicalEvidenceOccurrences: completeFixture.evidenceOccurrences.length,
      denseEvidenceOccurrences: completeFixture.evidenceOccurrences.filter(
        ({ embedding }) => embedding !== null,
      ).length,
      projectionRows: completeFixture.projections.length,
      embeddingDimensions: dimensions,
      productionMetadataReference,
    },
    lifecycle: {
      initialBuildMilliseconds,
      appendMilliseconds,
      optimizeMilliseconds: optimize.milliseconds,
      closeMilliseconds: initialClose.milliseconds + optimizedClose.milliseconds,
      reopenMilliseconds,
      replaySkippedRows: replay.skipped,
      closeReturnType: initialClose.returnType,
    },
    sizesAfterInitialBuild,
    sizesAfterAppend,
    sizesAfterOptimize,
    peakBytesObservedDuringOptimize: optimize.peakGenerationBytes,
    validation,
    neighborhood,
    replay: replayMeasurements,
    recoveryFaults,
    deletion,
    durabilityBoundary: {
      nativeDurabilityEstablished: false,
      lightweightPolicy:
        'Build outside the active generation, checkpoint bounded completed work, and record expected store counts and validation canaries in the manifest. On restart, reuse rows whose occurrence ID, embedding profile, and content checksum match, then embed only missing or mismatched rows. Keep the previous valid generation until replacement activation succeeds; rebuild the whole generation only when damage cannot be isolated.',
      optionalNativeCapability:
        'A status-returning flush or close and fail-open WAL recovery could reduce restart verification and re-embedding time. Adopt it only if zvec can expose it without adding a second recovery protocol or meaningful storage overhead.',
      deferredFaults: [
        'power-loss certification',
        'disk-full recovery for every internal write phase',
        'custom WAL repair',
        'native binary attestation',
      ],
    },
    verdict:
      'The three-store split is the simplest coherent topology, and its lookup, traversal, replay, sizing, validation, and resumable deletion protocols work on scratch evidence. Zvec cannot reliably certify its own crash recovery, but that does not block a rebuildable index. After a crash, verify and reuse valid embeddings, recompute only missing or damaged rows, and reserve a full rebuild for damage that cannot be isolated.',
  };
  const markdown = formatPrototypeReport(report);
  writeFileSync(MEASUREMENTS_PATH, markdown);
  writeAtomicJson(join(PROTOTYPE_ROOT, 'report.json'), report);
  return report;
}

function renderTerminalState(report: PrototypeReport | null): void {
  console.clear();
  console.log(`\u001b[1m${PROTOTYPE_NAME}\u001b[0m`);
  console.log('\u001b[2mScratch-only; never opens the live recall collection.\u001b[0m\n');
  if (report === null) {
    console.log('\u001b[1mCurrent state\u001b[0m');
    console.log(`report: ${existsSync(MEASUREMENTS_PATH) ? MEASUREMENTS_PATH : 'not run'}`);
    console.log(`scratch: ${existsSync(PROTOTYPE_ROOT) ? PROTOTYPE_ROOT : 'empty'}`);
  } else {
    console.log('\u001b[1mMeasured state\u001b[0m');
    console.log(`verdict: ${report.verdict}`);
    console.log(`lexical/source rows: ${report.validation.logicalLexicalSourceRows}`);
    console.log(`dense rows: ${report.validation.logicalDenseRows}`);
    console.log(`projection rows: ${report.validation.logicalProjectionRows}`);
    console.log(`close return: ${report.lifecycle.closeReturnType}`);
    console.log(`whole generation removed: ${report.deletion.wholeGenerationRemoved}`);
    console.log(`report: ${MEASUREMENTS_PATH}`);
  }
  console.log('\n\u001b[1mActions\u001b[0m');
  console.log('[a] run all probes  [v] view report  [w] wipe scratch  [q] quit');
}

async function runTerminalUi(): Promise<void> {
  const terminal = createInterface({ input: process.stdin, output: process.stdout });
  let report: PrototypeReport | null = null;
  try {
    while (true) {
      renderTerminalState(report);
      const action = (await terminal.question('> ')).trim().toLowerCase();
      if (action === 'q') {
        return;
      }
      if (action === 'a') {
        console.log('Running deterministic storage and fault probes…');
        report = await runPrototype();
      } else if (action === 'v') {
        console.clear();
        console.log(
          existsSync(MEASUREMENTS_PATH)
            ? readFileSync(MEASUREMENTS_PATH, 'utf8')
            : 'No report yet. Run [a] first.',
        );
        await terminal.question('\nPress Enter to return.');
      } else if (action === 'w') {
        rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
        report = null;
      }
    }
  } finally {
    terminal.close();
  }
}

const childWriterIndex = process.argv.indexOf('--child-crash-writer');
if (childWriterIndex >= 0) {
  const collectionPath = process.argv[childWriterIndex + 1];
  if (collectionPath === undefined) {
    throw new Error('Prototype crash writer path missing');
  }
  await runCrashWriterChild(collectionPath);
} else if (process.argv.includes('--all')) {
  const report = await runPrototype();
  console.log(JSON.stringify(report, null, 2));
  console.log(`\nMeasurements: ${MEASUREMENTS_PATH}`);
} else {
  await runTerminalUi();
}
