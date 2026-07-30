/* eslint-disable no-console -- this throwaway benchmark reports bounded progress to the terminal */

import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { chmod, cp, lstat, mkdir, readdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { ZVecOpen } from '@zvec/zvec';

import {
  createEmbeddingVectorCache,
  type EmbeddingVectorCache,
  type EmbeddingVectorCacheIdentity,
  normalizeConversationTextForEmbedding,
} from '../embedding-vector-cache.js';
import { RecallSessionProjectionKind, type SessionImportFormat } from '../enums.js';
import {
  createLocalEmbeddingClient,
  type LocalEmbeddingClient,
} from '../local-embedding-client.js';
import {
  loadOctenConversationTokenizer,
  OCTEN_TOKENIZER_IDENTITY,
} from '../octen-conversation-tokenizer.js';
import { loadRecallConversationConfig } from '../recall-conversation-config.js';
import type { RecallChunkPolicy } from '../recall-chunk-policy.js';
import { readNodeErrorCode } from '../read-node-error-code.js';
import {
  createLineageResolver,
  resolveProjectIdentity,
  type ResolvedProjectIdentity,
} from '../resolve-project-identity.js';
import type { ConversationTextTokenizer } from '../session-conversation-index.js';
import { openZvecSessionProjectionStore } from '../zvec-session-projection-store.js';
import {
  createRecallBenchmarkSplitStores,
  openRecallBenchmarkDenseTransferSource,
  type RecallBenchmarkDenseOccurrence,
  type RecallBenchmarkEvidenceOccurrence,
  type RecallBenchmarkSplitStores,
  type RecallBenchmarkStoreValidation,
} from './recall-benchmark-split-store.js';
import {
  createFrozenSizeStratifiedRecallSample,
  type FrozenRecallBenchmarkSample,
  type RecallBenchmarkSampleSummary,
} from './recall-benchmark-size-sample.js';
import {
  prepareRecallBenchmarkSource,
  readRecallBenchmarkEvidenceOccurrences,
} from './prepare-recall-benchmark-source.js';

const execFileAsync = promisify(execFile);
const BENCHMARK_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(BENCHMARK_DIRECTORY, '..', '..');
const PAGE_CACHE_EVICTION_SCRIPT = join(
  BENCHMARK_DIRECTORY,
  'evict_recall_benchmark_page_cache.py',
);
const SCRATCH_MARKER_FILE = '.pi-session-recall-no-cache-benchmark';
const FULL_SAMPLE_RATE = 0.1;
const PILOT_FILE_COUNT = 18;
const TARGET_EMBEDDING_DIMENSIONS = 1_024;
const NATIVE_OCTEN_DIMENSIONS = 2_560;
const EMBEDDING_REQUEST_BATCH_SIZE = 16;
const CACHE_RESOLUTION_BATCH_SIZE = 128;
const EMBEDDING_PROFILE_ID = 'octen-4b-prefix-l2-1024-issue-112-prototype-v1';
const CHUNK_POLICY: RecallChunkPolicy = { maxTokens: 512, overlapTokens: 64 };

type BenchmarkMode = 'pilot' | 'full';
type BenchmarkLaneName =
  | 'text_only'
  | 'cold_no_cache'
  | 'previous_generation_transfer'
  | 'warm_shared_cache';
type BenchmarkVectorMode = 'none' | 'model' | 'transfer' | 'cache';

interface BenchmarkArguments {
  mode: BenchmarkMode;
  scratchParent: string;
  keepScratch: boolean;
}

interface DirectorySizeMeasurement {
  files: number;
  apparentBytes: number;
  allocatedBytes: number;
}

interface GenerationSizeMeasurement {
  lexicalSource: DirectorySizeMeasurement;
  dense: DirectorySizeMeasurement;
  projections: DirectorySizeMeasurement;
  generation: DirectorySizeMeasurement;
}

interface BenchmarkLanePhases {
  storeCreationMilliseconds: number;
  importMilliseconds: number;
  projectionPreparationMilliseconds: number;
  projectIdentityMilliseconds: number;
  rowMaterializationMilliseconds: number;
  vectorResolutionMilliseconds: number;
  modelRequestMilliseconds: number;
  transferReadMilliseconds: number;
  cacheReadMilliseconds: number;
  lexicalWriteMilliseconds: number;
  denseWriteMilliseconds: number;
  projectionWriteMilliseconds: number;
  lexicalOptimizeMilliseconds: number;
  denseOptimizeMilliseconds: number;
  projectionOptimizeMilliseconds: number;
  closeMilliseconds: number;
  reopenValidationMilliseconds: number;
  unattributedMilliseconds: number;
  totalMilliseconds: number;
}

interface BenchmarkLaneCounts {
  scannedPhysicalSources: number;
  acceptedPhysicalSources: number;
  rejectedPhysicalSources: number;
  logicalSessions: number;
  entryAnchors: number;
  lexicalEvidenceOccurrences: number;
  denseEvidenceOccurrences: number;
  projectionRows: number;
  modelRequests: number;
  modelEmbeddedUniqueInputs: number;
  duplicateInputsReusedFromDenseStore: number;
  transferredVectors: number;
  cacheHits: number;
  cacheMisses: number;
}

interface BenchmarkFileMeasurement {
  stratumIndex: number;
  sourceBytes: number;
  accepted: boolean;
  rejectionFingerprint: string | null;
  format: SessionImportFormat | null;
  logicalSessions: number;
  entryAnchors: number;
  lexicalEvidenceOccurrences: number;
  denseEvidenceOccurrences: number;
  projectionRows: number;
  importMilliseconds: number;
  projectionPreparationMilliseconds: number;
  projectIdentityMilliseconds: number;
  rowMaterializationMilliseconds: number;
  vectorResolutionMilliseconds: number;
  lexicalWriteMilliseconds: number;
  denseWriteMilliseconds: number;
  projectionWriteMilliseconds: number;
  totalMilliseconds: number;
}

interface BenchmarkLaneValidation extends RecallBenchmarkStoreValidation {
  expectedProjectionRows: number;
  reopenedLexicalSourceRows: number;
  reopenedDenseRows: number;
  reopenedProjectionRows: number;
  physicalProjectionCount: number;
  reopenCountCanary: boolean;
}

interface BenchmarkLaneResult {
  lane: BenchmarkLaneName;
  vectorMode: BenchmarkVectorMode;
  phases: BenchmarkLanePhases;
  counts: BenchmarkLaneCounts;
  formatCounts: Record<string, number>;
  files: BenchmarkFileMeasurement[];
  sizesBeforeOptimize: GenerationSizeMeasurement;
  sizesAfterOptimize: GenerationSizeMeasurement;
  peakGenerationApparentBytesDuringOptimize: number;
  peakGenerationAllocatedBytesDuringOptimize: number;
  validation: BenchmarkLaneValidation;
  modelRequestLatencyQuantilesMilliseconds: Record<string, number>;
}

interface CacheSeedResult {
  sourceImportMilliseconds: number;
  sourceDenseReadMilliseconds: number;
  cacheResolutionMilliseconds: number;
  totalMilliseconds: number;
  cacheHits: number;
  cacheMissesWritten: number;
  fakeProviderRequests: number;
  cacheSize: DirectorySizeMeasurement;
}

interface RecallNoCacheBenchmarkResult {
  prototype: 'recall-no-persistent-embedding-cache-v1';
  mode: BenchmarkMode;
  generatedAt: string;
  gitRevision: string;
  model: {
    requestModel: string;
    servedModelId: string;
    nativeDimensions: number;
    storedDimensions: number;
    projection: 'first-n-then-l2-v1';
    requestConcurrency: 1;
    requestBatchSize: number;
  };
  chunkPolicy: RecallChunkPolicy;
  sample: RecallBenchmarkSampleSummary;
  sampleSourceFilesExcludedAsActive: number;
  pageCacheControl: {
    method: 'posix_fadvise-dontneed';
    sourceFilesEvictedBeforeEachLane: boolean;
    transferStoreEvictedBeforeRead: boolean;
    sharedCacheEvictedBeforeRead: boolean;
  };
  lanes: BenchmarkLaneResult[];
  cacheSeed: CacheSeedResult;
  comparisons: {
    coldNoCacheToTextOnlyTotalRatio: number;
    transferToWarmCacheTotalRatio: number;
    transferToWarmCacheVectorResolutionRatio: number;
    warmCacheToTransferVectorResolutionRatio: number;
    persistentCacheAllocatedBytesPerDenseOccurrence: number;
    persistentCacheAllocatedBytesPerUniqueEmbeddingInput: number;
    persistentCacheToDenseStoreAllocatedBytesRatio: number;
  };
  caveats: string[];
}

interface RunLaneOptions {
  lane: BenchmarkLaneName;
  vectorMode: BenchmarkVectorMode;
  generationDirectory: string;
  generationId: string;
  sample: FrozenRecallBenchmarkSample;
  tokenizer: ConversationTextTokenizer;
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>;
  nativeEmbeddingClient: LocalEmbeddingClient;
  transferSourceGenerationDirectory?: string;
  embeddingCache?: EmbeddingVectorCache;
  privateProgressPath: string;
}

function parseBenchmarkArguments(argumentsList: readonly string[]): BenchmarkArguments {
  const pilot = argumentsList.includes('--pilot');
  const full = argumentsList.includes('--full');
  if (pilot === full) {
    throw new Error('Recall no-cache benchmark requires exactly one of --pilot or --full');
  }
  let scratchParent = join(
    homedir(),
    '.cache',
    'pi-session-recall-prototypes',
    'issue-112-no-cache',
  );
  for (const [index, argument] of argumentsList.entries()) {
    if (argument.startsWith('--scratch-root=')) {
      scratchParent = argument.slice('--scratch-root='.length);
    } else if (argument === '--scratch-root') {
      const value = argumentsList[index + 1];
      if (!value) {
        throw new Error('Recall no-cache benchmark --scratch-root requires a path');
      }
      scratchParent = value;
    }
  }
  return {
    mode: pilot ? 'pilot' : 'full',
    scratchParent: resolve(scratchParent),
    keepScratch: argumentsList.includes('--keep-scratch'),
  };
}

function isPathAtOrBelow(path: string, root: string): boolean {
  const relativePath = relative(root, path);
  return relativePath === '' || (!relativePath.startsWith('..') && !relativePath.startsWith(sep));
}

async function createSafeScratchRunDirectory(options: {
  scratchParent: string;
  protectedRoots: readonly string[];
  mode: BenchmarkMode;
}): Promise<string> {
  for (const protectedRoot of options.protectedRoots) {
    if (
      isPathAtOrBelow(options.scratchParent, protectedRoot) ||
      isPathAtOrBelow(protectedRoot, options.scratchParent)
    ) {
      throw new Error(
        `Recall no-cache benchmark scratch parent overlaps a protected root: ${options.scratchParent}`,
      );
    }
  }
  await mkdir(options.scratchParent, { recursive: true, mode: 0o700 });
  await chmod(options.scratchParent, 0o700);
  const canonicalParent = await realpath(options.scratchParent);
  if (canonicalParent !== options.scratchParent) {
    throw new Error('Recall no-cache benchmark scratch parent must not traverse a symlink');
  }
  const runDirectory = join(options.scratchParent, `run-${options.mode}-${randomUUID()}`);
  await mkdir(runDirectory, { mode: 0o700 });
  await writeFile(join(runDirectory, SCRATCH_MARKER_FILE), 'private throwaway benchmark data\n', {
    encoding: 'utf8',
    mode: 0o600,
  });
  return runDirectory;
}

async function removeScratchRunDirectory(runDirectory: string): Promise<void> {
  const markerPath = join(runDirectory, SCRATCH_MARKER_FILE);
  const marker = await lstat(markerPath);
  if (!marker.isFile()) {
    throw new Error('Recall no-cache benchmark cleanup marker is not a file');
  }
  await rm(runDirectory, { recursive: true, force: false });
}

async function evictScratchPageCache(paths: readonly string[]): Promise<{
  files: number;
  bytes: number;
}> {
  const existingPaths = paths.filter((path) => existsSync(path));
  if (existingPaths.length === 0) {
    return { files: 0, bytes: 0 };
  }
  const { stdout } = await execFileAsync('uv', [
    'run',
    'python',
    PAGE_CACHE_EVICTION_SCRIPT,
    ...existingPaths,
  ]);
  const parsed: unknown = JSON.parse(stdout);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Recall no-cache benchmark page-cache control returned invalid JSON');
  }
  const files: unknown = Reflect.get(parsed, 'files');
  const bytes: unknown = Reflect.get(parsed, 'bytes');
  if (typeof files !== 'number' || typeof bytes !== 'number') {
    throw new Error('Recall no-cache benchmark page-cache control returned invalid counts');
  }
  return { files, bytes };
}

async function measureDirectory(path: string): Promise<DirectorySizeMeasurement> {
  if (!existsSync(path)) {
    return { files: 0, apparentBytes: 0, allocatedBytes: 0 };
  }
  let files = 0;
  let apparentBytes = 0;
  let allocatedBytes = 0;
  async function visit(current: string): Promise<void> {
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return;
      }
      throw error;
    }
    for (const entry of entries) {
      const entryPath = join(current, entry.name);
      if (entry.isDirectory()) {
        await visit(entryPath);
      } else if (entry.isFile()) {
        let metadata;
        try {
          metadata = await stat(entryPath, { bigint: true });
        } catch (error) {
          if (readNodeErrorCode(error) === 'ENOENT') {
            continue;
          }
          throw error;
        }
        files += 1;
        apparentBytes += Number(metadata.size);
        allocatedBytes += Number(metadata.blocks) * 512;
      }
    }
  }
  await visit(path);
  return { files, apparentBytes, allocatedBytes };
}

async function measureGeneration(generationDirectory: string): Promise<GenerationSizeMeasurement> {
  return {
    lexicalSource: await measureDirectory(join(generationDirectory, 'lexical-source')),
    dense: await measureDirectory(join(generationDirectory, 'dense')),
    projections: await measureDirectory(join(generationDirectory, 'projections')),
    generation: await measureDirectory(generationDirectory),
  };
}

function createEmptyPhases(): BenchmarkLanePhases {
  return {
    storeCreationMilliseconds: 0,
    importMilliseconds: 0,
    projectionPreparationMilliseconds: 0,
    projectIdentityMilliseconds: 0,
    rowMaterializationMilliseconds: 0,
    vectorResolutionMilliseconds: 0,
    modelRequestMilliseconds: 0,
    transferReadMilliseconds: 0,
    cacheReadMilliseconds: 0,
    lexicalWriteMilliseconds: 0,
    denseWriteMilliseconds: 0,
    projectionWriteMilliseconds: 0,
    lexicalOptimizeMilliseconds: 0,
    denseOptimizeMilliseconds: 0,
    projectionOptimizeMilliseconds: 0,
    closeMilliseconds: 0,
    reopenValidationMilliseconds: 0,
    unattributedMilliseconds: 0,
    totalMilliseconds: 0,
  };
}

function createEmptyCounts(): BenchmarkLaneCounts {
  return {
    scannedPhysicalSources: 0,
    acceptedPhysicalSources: 0,
    rejectedPhysicalSources: 0,
    logicalSessions: 0,
    entryAnchors: 0,
    lexicalEvidenceOccurrences: 0,
    denseEvidenceOccurrences: 0,
    projectionRows: 0,
    modelRequests: 0,
    modelEmbeddedUniqueInputs: 0,
    duplicateInputsReusedFromDenseStore: 0,
    transferredVectors: 0,
    cacheHits: 0,
    cacheMisses: 0,
  };
}

function fingerprintRejectedSource(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return createHash('sha256').update(message.replaceAll(REPOSITORY_ROOT, '<repo>')).digest('hex');
}

function projectOctenEmbeddingPrefix(nativeEmbedding: readonly number[]): number[] {
  if (nativeEmbedding.length !== NATIVE_OCTEN_DIMENSIONS) {
    throw new Error(
      `Recall no-cache benchmark expected ${NATIVE_OCTEN_DIMENSIONS} native dimensions, received ${nativeEmbedding.length}`,
    );
  }
  const prefix = nativeEmbedding.slice(0, TARGET_EMBEDDING_DIMENSIONS);
  const magnitude = Math.sqrt(prefix.reduce((sum, value) => sum + value * value, 0));
  if (!(magnitude > 0) || !Number.isFinite(magnitude)) {
    throw new Error('Recall no-cache benchmark Octen prefix has invalid L2 magnitude');
  }
  return prefix.map((value) => value / magnitude);
}

function assertProjectedEmbedding(vector: readonly number[]): void {
  if (
    vector.length !== TARGET_EMBEDDING_DIMENSIONS ||
    vector.some((value) => !Number.isFinite(value))
  ) {
    throw new Error('Recall no-cache benchmark projected embedding is invalid');
  }
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (Math.abs(magnitude - 1) > 1e-5) {
    throw new Error(
      `Recall no-cache benchmark projected embedding is not L2 normalized: ${magnitude}`,
    );
  }
}

function createCachedProjectIdentityResolver(
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
): (workingDirectory: string) => Promise<ResolvedProjectIdentity | null> {
  const resolveLineage = createLineageResolver(config.projectLineages, resolveProjectIdentity);
  const resolutions = new Map<string, Promise<ResolvedProjectIdentity | null>>();
  return (workingDirectory) => {
    const existing = resolutions.get(workingDirectory);
    if (existing) {
      return existing;
    }
    const resolution = resolveLineage(workingDirectory);
    resolutions.set(workingDirectory, resolution);
    return resolution;
  };
}

function createEmbeddingCacheIdentity(
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
): EmbeddingVectorCacheIdentity {
  return {
    embedding: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: TARGET_EMBEDDING_DIMENSIONS,
      quantization: config.embeddingQuantization,
      pooling: config.embeddingPooling,
      normalization: 'l2',
      canaryProbe: 'issue 112 no-cache benchmark profile canary',
      canaryFingerprint: createHash('sha256').update(EMBEDDING_PROFILE_ID).digest('hex'),
    },
    tokenizer: {
      model: OCTEN_TOKENIZER_IDENTITY.model,
      revision: OCTEN_TOKENIZER_IDENTITY.revision,
      library: { ...OCTEN_TOKENIZER_IDENTITY.library },
      encodeOptions: { ...OCTEN_TOKENIZER_IDENTITY.encodeOptions },
      assets: [
        {
          fileName: OCTEN_TOKENIZER_IDENTITY.tokenizerJson.fileName,
          sha256: OCTEN_TOKENIZER_IDENTITY.tokenizerJson.sha256,
        },
        {
          fileName: OCTEN_TOKENIZER_IDENTITY.tokenizerConfigJson.fileName,
          sha256: OCTEN_TOKENIZER_IDENTITY.tokenizerConfigJson.sha256,
        },
      ],
    },
    chunkPolicy: { version: 2, normalization: 'unicode-nfc-v1' },
  };
}

function quantiles(values: readonly number[]): Record<string, number> {
  if (values.length === 0) {
    return { minimum: 0, p50: 0, p90: 0, p95: 0, p99: 0, maximum: 0 };
  }
  const sorted = [...values].sort((left, right) => left - right);
  const at = (proportion: number): number =>
    sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * proportion))] ?? 0;
  return {
    minimum: at(0),
    p50: at(0.5),
    p90: at(0.9),
    p95: at(0.95),
    p99: at(0.99),
    maximum: at(1),
  };
}

async function resolveColdNoCacheVectors(options: {
  occurrences: readonly RecallBenchmarkEvidenceOccurrence[];
  targetStores: RecallBenchmarkSplitStores;
  nativeEmbeddingClient: LocalEmbeddingClient;
  seenOccurrenceIdByEmbeddingInputChecksum: Map<string, string>;
  counts: BenchmarkLaneCounts;
  phases: BenchmarkLanePhases;
  requestLatencies: number[];
}): Promise<RecallBenchmarkDenseOccurrence[]> {
  const vectorByInputChecksum = new Map<string, number[]>();
  const previousOccurrenceByInputChecksum = new Map<string, string>();
  for (const occurrence of options.occurrences) {
    const previousOccurrenceId = options.seenOccurrenceIdByEmbeddingInputChecksum.get(
      occurrence.embeddingInputChecksum,
    );
    if (previousOccurrenceId) {
      previousOccurrenceByInputChecksum.set(
        occurrence.embeddingInputChecksum,
        previousOccurrenceId,
      );
    }
  }
  const previousRows = options.targetStores.fetchDenseEvidence(
    [...new Set(previousOccurrenceByInputChecksum.values())].map((occurrenceId) => ({
      occurrenceId,
    })),
  );
  for (const [inputChecksum, occurrenceId] of previousOccurrenceByInputChecksum) {
    const previous = previousRows.get(occurrenceId);
    if (
      !previous ||
      previous.embeddingInputChecksum !== inputChecksum ||
      previous.embeddingProfileId !== EMBEDDING_PROFILE_ID
    ) {
      throw new Error('Recall no-cache benchmark dense duplicate reuse verification failed');
    }
    vectorByInputChecksum.set(inputChecksum, previous.embedding);
  }

  const uniqueModelInputs = new Map<string, RecallBenchmarkEvidenceOccurrence>();
  for (const occurrence of options.occurrences) {
    if (!vectorByInputChecksum.has(occurrence.embeddingInputChecksum)) {
      uniqueModelInputs.set(occurrence.embeddingInputChecksum, occurrence);
    }
  }
  const inputs = [...uniqueModelInputs.values()];
  for (let start = 0; start < inputs.length; start += EMBEDDING_REQUEST_BATCH_SIZE) {
    const batch = inputs.slice(start, start + EMBEDDING_REQUEST_BATCH_SIZE);
    const normalizedTexts = batch.map(({ chunk }) =>
      normalizeConversationTextForEmbedding(chunk.content),
    );
    const requestStartedAt = performance.now();
    const nativeEmbeddings = await options.nativeEmbeddingClient.embedTexts(normalizedTexts);
    const requestMilliseconds = performance.now() - requestStartedAt;
    options.requestLatencies.push(requestMilliseconds);
    options.phases.modelRequestMilliseconds += requestMilliseconds;
    options.counts.modelRequests += 1;
    options.counts.modelEmbeddedUniqueInputs += batch.length;
    for (const [index, occurrence] of batch.entries()) {
      const nativeEmbedding = nativeEmbeddings[index];
      if (!nativeEmbedding) {
        throw new Error('Recall no-cache benchmark model omitted an embedding');
      }
      const embedding = projectOctenEmbeddingPrefix(nativeEmbedding);
      assertProjectedEmbedding(embedding);
      vectorByInputChecksum.set(occurrence.embeddingInputChecksum, embedding);
    }
  }
  options.counts.duplicateInputsReusedFromDenseStore += options.occurrences.length - inputs.length;
  return options.occurrences.map((occurrence) => {
    const embedding = vectorByInputChecksum.get(occurrence.embeddingInputChecksum);
    if (!embedding) {
      throw new Error('Recall no-cache benchmark cold vector resolution is incomplete');
    }
    return { occurrence, embeddingProfileId: EMBEDDING_PROFILE_ID, embedding };
  });
}

function resolveTransferredVectors(options: {
  occurrences: readonly RecallBenchmarkEvidenceOccurrence[];
  transferSource: Pick<RecallBenchmarkSplitStores, 'fetchDenseEvidence'>;
  counts: BenchmarkLaneCounts;
}): RecallBenchmarkDenseOccurrence[] {
  const recovered = options.transferSource.fetchDenseEvidence(options.occurrences);
  return options.occurrences.map((occurrence) => {
    const previous = recovered.get(occurrence.occurrenceId);
    if (
      !previous ||
      previous.checksum !== occurrence.chunk.checksum ||
      previous.embeddingInputChecksum !== occurrence.embeddingInputChecksum ||
      previous.embeddingProfileId !== EMBEDDING_PROFILE_ID
    ) {
      throw new Error(
        'Recall no-cache benchmark previous-generation vector transfer verification failed',
      );
    }
    options.counts.transferredVectors += 1;
    return {
      occurrence,
      embeddingProfileId: EMBEDDING_PROFILE_ID,
      embedding: previous.embedding,
    };
  });
}

async function resolveWarmCacheVectors(options: {
  occurrences: readonly RecallBenchmarkEvidenceOccurrence[];
  embeddingCache: EmbeddingVectorCache;
  counts: BenchmarkLaneCounts;
  phases: BenchmarkLanePhases;
}): Promise<RecallBenchmarkDenseOccurrence[]> {
  const vectors: number[][] = [];
  for (let start = 0; start < options.occurrences.length; start += CACHE_RESOLUTION_BATCH_SIZE) {
    const batch = options.occurrences.slice(start, start + CACHE_RESOLUTION_BATCH_SIZE);
    const result = await options.embeddingCache.resolveEmbeddingVectors(
      batch.map(({ chunk }) => chunk.content),
    );
    options.counts.cacheHits += result.cacheHits;
    options.counts.cacheMisses += result.newlyEmbeddedChunks;
    options.phases.cacheReadMilliseconds += result.embeddingCacheResolutionMilliseconds;
    if (result.newlyEmbeddedChunks !== 0 || result.embeddingRequestCount !== 0) {
      throw new Error('Recall no-cache benchmark warm cache unexpectedly missed');
    }
    vectors.push(...result.vectors);
  }
  return options.occurrences.map((occurrence, index) => {
    const embedding = vectors[index];
    if (!embedding) {
      throw new Error('Recall no-cache benchmark warm cache omitted a vector');
    }
    return { occurrence, embeddingProfileId: EMBEDDING_PROFILE_ID, embedding };
  });
}

async function monitorPeakGenerationSize<T>(
  generationDirectory: string,
  operation: () => Promise<T>,
): Promise<{
  result: T;
  peakApparentBytes: number;
  peakAllocatedBytes: number;
}> {
  let peakApparentBytes = 0;
  let peakAllocatedBytes = 0;
  let sampling = false;
  const sample = async (): Promise<void> => {
    if (sampling) {
      return;
    }
    sampling = true;
    try {
      const size = await measureDirectory(generationDirectory);
      peakApparentBytes = Math.max(peakApparentBytes, size.apparentBytes);
      peakAllocatedBytes = Math.max(peakAllocatedBytes, size.allocatedBytes);
    } finally {
      sampling = false;
    }
  };
  await sample();
  const interval = setInterval(() => void sample(), 100);
  try {
    const result = await operation();
    await sample();
    return { result, peakApparentBytes, peakAllocatedBytes };
  } finally {
    clearInterval(interval);
  }
}

async function writePrivateProgress(path: string, value: Record<string, unknown>): Promise<void> {
  await writeFile(path, `${JSON.stringify(value)}\n`, { encoding: 'utf8', mode: 0o600 });
}

async function runBenchmarkLane(options: RunLaneOptions): Promise<BenchmarkLaneResult> {
  console.log(`\n[${options.lane}] evicting scratch source pages`);
  await evictScratchPageCache([join(dirname(options.sample.privateManifestPath))]);
  if (options.transferSourceGenerationDirectory) {
    await evictScratchPageCache([join(options.transferSourceGenerationDirectory, 'dense')]);
  }
  const laneStartedAt = performance.now();
  const phases = createEmptyPhases();
  const counts = createEmptyCounts();
  const formatCounts: Record<string, number> = {};
  const files: BenchmarkFileMeasurement[] = [];
  const requestLatencies: number[] = [];
  const seenOccurrenceIdByEmbeddingInputChecksum = new Map<string, string>();
  const resolveProject = createCachedProjectIdentityResolver(options.config);

  const storeCreationStartedAt = performance.now();
  const stores = createRecallBenchmarkSplitStores({
    generationDirectory: options.generationDirectory,
    embeddingDimensions: TARGET_EMBEDDING_DIMENSIONS,
  });
  const projectionStore = openZvecSessionProjectionStore({
    databasePath: join(options.generationDirectory, 'projections'),
    generationId: options.generationId,
    createIfMissing: true,
    readOnly: false,
  });
  phases.storeCreationMilliseconds += performance.now() - storeCreationStartedAt;

  const transferSource = options.transferSourceGenerationDirectory
    ? openRecallBenchmarkDenseTransferSource(options.transferSourceGenerationDirectory)
    : null;
  let anchorCanaryId: string | null = null;
  let evidenceCanary: RecallBenchmarkEvidenceOccurrence | null = null;
  let denseCanary: RecallBenchmarkDenseOccurrence | null = null;
  let expectedLexicalSourceRows = 0;
  let expectedDenseRows = 0;
  let expectedProjectionRows = 0;

  try {
    for (const [fileIndex, file] of options.sample.files.entries()) {
      counts.scannedPhysicalSources += 1;
      const fileStartedAt = performance.now();
      let prepared;
      try {
        prepared = await prepareRecallBenchmarkSource({
          physicalSessionPath: file.snapshotPath,
          relativePath: file.relativePath,
          expectedSourceBytes: file.snapshotBytes,
          generationId: options.generationId,
          tokenizer: options.tokenizer,
          chunkPolicy: CHUNK_POLICY,
          resolveProjectIdentity: resolveProject,
        });
      } catch (error) {
        counts.rejectedPhysicalSources += 1;
        files.push({
          stratumIndex: file.stratumIndex,
          sourceBytes: file.snapshotBytes,
          accepted: false,
          rejectionFingerprint: fingerprintRejectedSource(error),
          format: null,
          logicalSessions: 0,
          entryAnchors: 0,
          lexicalEvidenceOccurrences: 0,
          denseEvidenceOccurrences: 0,
          projectionRows: 0,
          importMilliseconds: 0,
          projectionPreparationMilliseconds: 0,
          projectIdentityMilliseconds: 0,
          rowMaterializationMilliseconds: 0,
          vectorResolutionMilliseconds: 0,
          lexicalWriteMilliseconds: 0,
          denseWriteMilliseconds: 0,
          projectionWriteMilliseconds: 0,
          totalMilliseconds: performance.now() - fileStartedAt,
        });
        continue;
      }

      counts.acceptedPhysicalSources += 1;
      counts.logicalSessions += prepared.projections.filter(
        ({ projectionKind }) => projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
      ).length;
      counts.entryAnchors += prepared.entryAnchors.length;
      counts.lexicalEvidenceOccurrences += prepared.evidenceOccurrences.length;
      counts.denseEvidenceOccurrences += prepared.denseOccurrences.length;
      counts.projectionRows += prepared.projections.length;
      formatCounts[prepared.format] = (formatCounts[prepared.format] ?? 0) + 1;
      phases.importMilliseconds += prepared.timings.importMilliseconds;
      phases.projectionPreparationMilliseconds += prepared.timings.projectionMilliseconds;
      phases.projectIdentityMilliseconds += prepared.timings.projectIdentityMilliseconds;
      phases.rowMaterializationMilliseconds += prepared.timings.rowMaterializationMilliseconds;

      const vectorResolutionStartedAt = performance.now();
      let denseRows: RecallBenchmarkDenseOccurrence[] = [];
      if (options.vectorMode === 'model') {
        denseRows = await resolveColdNoCacheVectors({
          occurrences: prepared.denseOccurrences,
          targetStores: stores,
          nativeEmbeddingClient: options.nativeEmbeddingClient,
          seenOccurrenceIdByEmbeddingInputChecksum,
          counts,
          phases,
          requestLatencies,
        });
      } else if (options.vectorMode === 'transfer') {
        if (!transferSource) {
          throw new Error('Recall no-cache benchmark transfer lane has no source generation');
        }
        const transferReadStartedAt = performance.now();
        denseRows = resolveTransferredVectors({
          occurrences: prepared.denseOccurrences,
          transferSource,
          counts,
        });
        phases.transferReadMilliseconds += performance.now() - transferReadStartedAt;
      } else if (options.vectorMode === 'cache') {
        if (!options.embeddingCache) {
          throw new Error('Recall no-cache benchmark cache lane has no cache');
        }
        denseRows = await resolveWarmCacheVectors({
          occurrences: prepared.denseOccurrences,
          embeddingCache: options.embeddingCache,
          counts,
          phases,
        });
      }
      const vectorResolutionMilliseconds = performance.now() - vectorResolutionStartedAt;
      phases.vectorResolutionMilliseconds += vectorResolutionMilliseconds;

      const lexicalWriteStartedAt = performance.now();
      stores.insertEntryAnchors(prepared.entryAnchors);
      stores.insertLexicalEvidence(prepared.evidenceOccurrences);
      const lexicalWriteMilliseconds = performance.now() - lexicalWriteStartedAt;
      phases.lexicalWriteMilliseconds += lexicalWriteMilliseconds;

      const denseWriteStartedAt = performance.now();
      stores.insertDenseEvidence(denseRows);
      const denseWriteMilliseconds = performance.now() - denseWriteStartedAt;
      phases.denseWriteMilliseconds += denseWriteMilliseconds;
      if (options.vectorMode === 'model') {
        for (const occurrence of prepared.denseOccurrences) {
          if (!seenOccurrenceIdByEmbeddingInputChecksum.has(occurrence.embeddingInputChecksum)) {
            seenOccurrenceIdByEmbeddingInputChecksum.set(
              occurrence.embeddingInputChecksum,
              occurrence.occurrenceId,
            );
          }
        }
      }

      const projectionWriteStartedAt = performance.now();
      await projectionStore.upsertProjections(prepared.projections);
      const projectionWriteMilliseconds = performance.now() - projectionWriteStartedAt;
      phases.projectionWriteMilliseconds += projectionWriteMilliseconds;

      anchorCanaryId ??= prepared.entryAnchors[0]?.anchorId ?? null;
      evidenceCanary ??= prepared.evidenceOccurrences[0] ?? null;
      denseCanary ??= denseRows[0] ?? null;
      expectedLexicalSourceRows +=
        prepared.entryAnchors.length + prepared.evidenceOccurrences.length;
      expectedDenseRows += denseRows.length;
      expectedProjectionRows += prepared.projections.length;
      files.push({
        stratumIndex: file.stratumIndex,
        sourceBytes: file.snapshotBytes,
        accepted: true,
        rejectionFingerprint: null,
        format: prepared.format,
        logicalSessions: prepared.projections.filter(
          ({ projectionKind }) => projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
        ).length,
        entryAnchors: prepared.entryAnchors.length,
        lexicalEvidenceOccurrences: prepared.evidenceOccurrences.length,
        denseEvidenceOccurrences: prepared.denseOccurrences.length,
        projectionRows: prepared.projections.length,
        importMilliseconds: prepared.timings.importMilliseconds,
        projectionPreparationMilliseconds: prepared.timings.projectionMilliseconds,
        projectIdentityMilliseconds: prepared.timings.projectIdentityMilliseconds,
        rowMaterializationMilliseconds: prepared.timings.rowMaterializationMilliseconds,
        vectorResolutionMilliseconds,
        lexicalWriteMilliseconds,
        denseWriteMilliseconds,
        projectionWriteMilliseconds,
        totalMilliseconds: performance.now() - fileStartedAt,
      });

      if ((fileIndex + 1) % 5 === 0 || fileIndex + 1 === options.sample.files.length) {
        console.log(
          `[${options.lane}] ${fileIndex + 1}/${options.sample.files.length} sources; ` +
            `${counts.denseEvidenceOccurrences} dense rows; ${counts.modelRequests} model requests`,
        );
        await writePrivateProgress(options.privateProgressPath, {
          lane: options.lane,
          completedSources: fileIndex + 1,
          totalSources: options.sample.files.length,
          denseEvidenceOccurrences: counts.denseEvidenceOccurrences,
          modelRequests: counts.modelRequests,
          elapsedMilliseconds: performance.now() - laneStartedAt,
        });
      }
    }

    const sizesBeforeOptimize = await measureGeneration(options.generationDirectory);
    projectionStore.close();
    const optimized = await monitorPeakGenerationSize(options.generationDirectory, async () => {
      const storeOptimization = await stores.optimize();
      phases.lexicalOptimizeMilliseconds += storeOptimization.lexicalSourceMilliseconds;
      phases.denseOptimizeMilliseconds += storeOptimization.denseMilliseconds;
      const projectionOptimizeStartedAt = performance.now();
      const projectionCollection = ZVecOpen(join(options.generationDirectory, 'projections'));
      try {
        if (projectionCollection.stats.docCount > 0) {
          await projectionCollection.optimize();
        }
      } finally {
        projectionCollection.closeSync();
      }
      phases.projectionOptimizeMilliseconds += performance.now() - projectionOptimizeStartedAt;
    });
    const sizesAfterOptimize = await measureGeneration(options.generationDirectory);
    const storeValidation = stores.validate({
      expectedLexicalSourceRows,
      expectedDenseRows,
      anchorCanaryId,
      evidenceCanary,
      denseCanary,
    });

    const closeStartedAt = performance.now();
    stores.close();
    transferSource?.close();
    phases.closeMilliseconds += performance.now() - closeStartedAt;

    const reopenStartedAt = performance.now();
    const reopenedLexical = ZVecOpen(join(options.generationDirectory, 'lexical-source'), {
      readOnly: true,
    });
    const reopenedDense = ZVecOpen(join(options.generationDirectory, 'dense'), { readOnly: true });
    const reopenedProjections = openZvecSessionProjectionStore({
      databasePath: join(options.generationDirectory, 'projections'),
      generationId: options.generationId,
      createIfMissing: false,
      readOnly: true,
    });
    const reopenedLexicalSourceRows = reopenedLexical.stats.docCount;
    const reopenedDenseRows = reopenedDense.stats.docCount;
    const physicalProjectionCount = reopenedProjections.listPhysicalProjections().length;
    const projectionCollection = ZVecOpen(join(options.generationDirectory, 'projections'), {
      readOnly: true,
    });
    const reopenedProjectionRows = projectionCollection.stats.docCount;
    projectionCollection.closeSync();
    reopenedLexical.closeSync();
    reopenedDense.closeSync();
    reopenedProjections.close();
    phases.reopenValidationMilliseconds += performance.now() - reopenStartedAt;

    phases.totalMilliseconds = performance.now() - laneStartedAt;
    const {
      totalMilliseconds,
      unattributedMilliseconds: ignoredUnattributedMilliseconds,
      ...attributedPhases
    } = phases;
    void ignoredUnattributedMilliseconds;
    const attributedMilliseconds = Object.values(attributedPhases).reduce(
      (total, milliseconds) => total + milliseconds,
      0,
    );
    phases.unattributedMilliseconds = Math.max(totalMilliseconds - attributedMilliseconds, 0);
    const validation: BenchmarkLaneValidation = {
      ...storeValidation,
      expectedProjectionRows,
      reopenedLexicalSourceRows,
      reopenedDenseRows,
      reopenedProjectionRows,
      physicalProjectionCount,
      reopenCountCanary:
        reopenedLexicalSourceRows === expectedLexicalSourceRows &&
        reopenedDenseRows === expectedDenseRows &&
        reopenedProjectionRows === expectedProjectionRows &&
        physicalProjectionCount === counts.acceptedPhysicalSources,
    };
    return {
      lane: options.lane,
      vectorMode: options.vectorMode,
      phases,
      counts,
      formatCounts,
      files,
      sizesBeforeOptimize,
      sizesAfterOptimize,
      peakGenerationApparentBytesDuringOptimize: optimized.peakApparentBytes,
      peakGenerationAllocatedBytesDuringOptimize: optimized.peakAllocatedBytes,
      validation,
      modelRequestLatencyQuantilesMilliseconds: quantiles(requestLatencies),
    };
  } catch (error) {
    try {
      projectionStore.close();
    } catch {
      // The original benchmark failure remains primary.
    }
    try {
      stores.close();
    } catch {
      // The original benchmark failure remains primary.
    }
    try {
      transferSource?.close();
    } catch {
      // The original benchmark failure remains primary.
    }
    throw error;
  }
}

async function seedScratchEmbeddingCache(options: {
  sample: FrozenRecallBenchmarkSample;
  acceptedStrata: ReadonlySet<number>;
  sourceGenerationDirectory: string;
  cacheDirectory: string;
  tokenizer: ConversationTextTokenizer;
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>;
}): Promise<{ cache: EmbeddingVectorCache; result: CacheSeedResult }> {
  console.log('\n[cache_seed] evicting scratch source and dense pages');
  await evictScratchPageCache([
    dirname(options.sample.privateManifestPath),
    join(options.sourceGenerationDirectory, 'dense'),
  ]);
  const source = openRecallBenchmarkDenseTransferSource(options.sourceGenerationDirectory);
  let activeVectorsByInputChecksum = new Map<string, number[]>();
  let fakeProviderRequests = 0;
  const fakeEmbeddingProvider: LocalEmbeddingClient = {
    async embedTexts(texts) {
      fakeProviderRequests += 1;
      return texts.map((text) => {
        const checksum = createHash('sha256')
          .update(normalizeConversationTextForEmbedding(text))
          .digest('hex');
        const vector = activeVectorsByInputChecksum.get(checksum);
        if (!vector) {
          throw new Error('Recall no-cache benchmark cache seed vector is missing');
        }
        return vector;
      });
    },
  };
  const cache = createEmbeddingVectorCache({
    cacheDirectory: options.cacheDirectory,
    identity: createEmbeddingCacheIdentity(options.config),
    embeddingRequestBatchSize: EMBEDDING_REQUEST_BATCH_SIZE,
    embeddings: fakeEmbeddingProvider,
  });
  const startedAt = performance.now();
  let sourceImportMilliseconds = 0;
  let sourceDenseReadMilliseconds = 0;
  let cacheResolutionMilliseconds = 0;
  let cacheHits = 0;
  let cacheMissesWritten = 0;
  try {
    for (const file of options.sample.files) {
      if (!options.acceptedStrata.has(file.stratumIndex)) {
        continue;
      }
      const importStartedAt = performance.now();
      const imported = await readRecallBenchmarkEvidenceOccurrences({
        physicalSessionPath: file.snapshotPath,
        relativePath: file.relativePath,
        tokenizer: options.tokenizer,
        chunkPolicy: CHUNK_POLICY,
      });
      sourceImportMilliseconds += performance.now() - importStartedAt;
      for (
        let start = 0;
        start < imported.denseOccurrences.length;
        start += CACHE_RESOLUTION_BATCH_SIZE
      ) {
        const batch = imported.denseOccurrences.slice(start, start + CACHE_RESOLUTION_BATCH_SIZE);
        const sourceReadStartedAt = performance.now();
        const recovered = source.fetchDenseEvidence(batch);
        sourceDenseReadMilliseconds += performance.now() - sourceReadStartedAt;
        activeVectorsByInputChecksum = new Map();
        for (const occurrence of batch) {
          const previous = recovered.get(occurrence.occurrenceId);
          if (
            !previous ||
            previous.checksum !== occurrence.chunk.checksum ||
            previous.embeddingInputChecksum !== occurrence.embeddingInputChecksum ||
            previous.embeddingProfileId !== EMBEDDING_PROFILE_ID
          ) {
            throw new Error('Recall no-cache benchmark cache seed source verification failed');
          }
          activeVectorsByInputChecksum.set(occurrence.embeddingInputChecksum, previous.embedding);
        }
        const cacheStartedAt = performance.now();
        const result = await cache.resolveEmbeddingVectors(batch.map(({ chunk }) => chunk.content));
        cacheResolutionMilliseconds += performance.now() - cacheStartedAt;
        cacheHits += result.cacheHits;
        cacheMissesWritten += result.newlyEmbeddedChunks;
      }
    }
  } finally {
    source.close();
  }
  const totalMilliseconds = performance.now() - startedAt;
  return {
    cache,
    result: {
      sourceImportMilliseconds,
      sourceDenseReadMilliseconds,
      cacheResolutionMilliseconds,
      totalMilliseconds,
      cacheHits,
      cacheMissesWritten,
      fakeProviderRequests,
      cacheSize: await measureDirectory(options.cacheDirectory),
    },
  };
}

function assertBenchmarkLaneEquivalence(lanes: readonly BenchmarkLaneResult[]): void {
  const baseline = lanes[0];
  if (!baseline) {
    throw new Error('Recall no-cache benchmark produced no lanes');
  }
  const baselineAcceptedStrata = baseline.files
    .filter(({ accepted }) => accepted)
    .map(({ stratumIndex }) => stratumIndex)
    .join(',');
  for (const lane of lanes.slice(1)) {
    const acceptedStrata = lane.files
      .filter(({ accepted }) => accepted)
      .map(({ stratumIndex }) => stratumIndex)
      .join(',');
    if (
      acceptedStrata !== baselineAcceptedStrata ||
      lane.counts.entryAnchors !== baseline.counts.entryAnchors ||
      lane.counts.lexicalEvidenceOccurrences !== baseline.counts.lexicalEvidenceOccurrences ||
      lane.counts.denseEvidenceOccurrences !== baseline.counts.denseEvidenceOccurrences ||
      lane.counts.projectionRows !== baseline.counts.projectionRows
    ) {
      throw new Error(`Recall no-cache benchmark lane ${lane.lane} is not source-equivalent`);
    }
  }
  const coldLane = lanes.find(({ lane }) => lane === 'cold_no_cache');
  if (!coldLane || coldLane.counts.modelRequests < 1) {
    throw new Error('Recall no-cache benchmark cold lane made no model requests');
  }
  if (lanes.some(({ lane, counts }) => lane !== 'cold_no_cache' && counts.modelRequests !== 0)) {
    throw new Error('Recall no-cache benchmark non-cold lane made a model request');
  }
}

function safeRatio(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

function formatMilliseconds(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${milliseconds.toFixed(1)} ms`;
  }
  const seconds = milliseconds / 1_000;
  if (seconds < 60) {
    return `${seconds.toFixed(2)} s`;
  }
  const minutes = seconds / 60;
  return minutes < 60 ? `${minutes.toFixed(2)} min` : `${(minutes / 60).toFixed(2)} h`;
}

function formatBytes(bytes: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes;
  let unit = units[0] ?? 'B';
  for (const candidate of units) {
    unit = candidate;
    if (value < 1_024 || candidate === units.at(-1)) {
      break;
    }
    value /= 1_024;
  }
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${unit}`;
}

function validationPassed(validation: BenchmarkLaneValidation): boolean {
  return Object.entries(validation)
    .filter(([name]) => name.endsWith('Canary'))
    .every(([, value]) => value === true);
}

function formatBenchmarkReport(result: RecallNoCacheBenchmarkResult): string {
  const laneRows = result.lanes
    .map(
      (lane) =>
        `| ${lane.lane} | ${formatMilliseconds(lane.phases.totalMilliseconds)} | ${formatMilliseconds(lane.phases.importMilliseconds)} | ${formatMilliseconds(lane.phases.projectionPreparationMilliseconds)} | ${formatMilliseconds(lane.phases.vectorResolutionMilliseconds)} | ${formatMilliseconds(lane.phases.lexicalWriteMilliseconds)} | ${formatMilliseconds(lane.phases.denseWriteMilliseconds)} | ${formatMilliseconds(lane.phases.lexicalOptimizeMilliseconds + lane.phases.denseOptimizeMilliseconds + lane.phases.projectionOptimizeMilliseconds)} |`,
    )
    .join('\n');
  const sizeRows = result.lanes
    .map(
      (lane) =>
        `| ${lane.lane} | ${formatBytes(lane.sizesAfterOptimize.lexicalSource.allocatedBytes)} | ${formatBytes(lane.sizesAfterOptimize.dense.allocatedBytes)} | ${formatBytes(lane.sizesAfterOptimize.projections.allocatedBytes)} | ${formatBytes(lane.sizesAfterOptimize.generation.allocatedBytes)} | ${formatBytes(lane.peakGenerationAllocatedBytesDuringOptimize)} |`,
    )
    .join('\n');
  const countRows = result.lanes
    .map(
      (lane) =>
        `| ${lane.lane} | ${lane.counts.acceptedPhysicalSources} | ${lane.counts.logicalSessions} | ${lane.counts.entryAnchors} | ${lane.counts.lexicalEvidenceOccurrences} | ${lane.counts.denseEvidenceOccurrences} | ${lane.counts.modelRequests} | ${lane.counts.transferredVectors} | ${lane.counts.cacheHits} |`,
    )
    .join('\n');
  const validationRows = result.lanes
    .map(
      (lane) =>
        `| ${lane.lane} | ${validationPassed(lane.validation)} | ${lane.validation.lexicalSourceRows} | ${lane.validation.denseRows} | ${lane.validation.reopenedProjectionRows} | ${lane.validation.physicalProjectionCount} |`,
    )
    .join('\n');
  const formatRows = Object.entries(result.lanes[0]?.formatCounts ?? {})
    .map(([format, count]) => `| ${format} | ${count} |`)
    .join('\n');
  return `# PROTOTYPE measurements — rebuild recall without a persistent embedding cache

Generated ${result.generatedAt} from commit \`${result.gitRevision}\` in \`${result.mode}\` mode.

## Question

Under the proposed split-store generation, does a persistent embedding cache save enough time to justify its extra disk use, file count, and lifecycle?

This report does not apply a fixed acceptable-duration threshold. It compares relative phase time and storage on one reproducible size-stratified sample.

## Sample

- Physical source population: ${result.sample.populationFiles.toLocaleString()} files / ${formatBytes(result.sample.populationBytes)}
- Selected sample: ${result.sample.selectedFiles.toLocaleString()} files / ${formatBytes(result.sample.selectedBytes)}
- File sampling rate: ${(result.sample.selectedFileRate * 100).toFixed(2)}%
- Byte sampling rate: ${(result.sample.selectedByteRate * 100).toFixed(2)}%
- Sample identity: \`${result.sample.sampleIdentitySha256}\`
- Seed: \`${result.sample.seed}\`

| Imported physical format | Accepted files |
| --- | ---: |
${formatRows}

## Lane time

| Lane | Total | Import | Projection preparation | Vector resolution | Lexical write | Dense write | All optimize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${laneRows}

## Logical work

| Lane | Accepted sources | Logical sessions | Anchors | Lexical evidence | Dense evidence | Model requests | Transferred vectors | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${countRows}

## Allocated storage after optimize

| Lane | Lexical/source | Dense | Projections | Whole generation | Peak generation during optimize |
| --- | ---: | ---: | ---: | ---: | ---: |
${sizeRows}

Shared cache after complete seed: ${formatBytes(result.cacheSeed.cacheSize.allocatedBytes)} across ${result.cacheSeed.cacheSize.files.toLocaleString()} files.

Cache seeding itself took ${formatMilliseconds(result.cacheSeed.totalMilliseconds)} total, including ${formatMilliseconds(result.cacheSeed.sourceImportMilliseconds)} source import, ${formatMilliseconds(result.cacheSeed.sourceDenseReadMilliseconds)} dense reads, and ${formatMilliseconds(result.cacheSeed.cacheResolutionMilliseconds)} production-format cache resolution/writes.

## Relative comparisons

- Cold no-cache total / text-only total: **${result.comparisons.coldNoCacheToTextOnlyTotalRatio.toFixed(3)}×**
- Previous-generation transfer total / warm shared-cache total: **${result.comparisons.transferToWarmCacheTotalRatio.toFixed(3)}×**
- Previous-generation transfer vector resolution / warm-cache vector resolution: **${result.comparisons.transferToWarmCacheVectorResolutionRatio.toFixed(3)}×**
- Warm-cache vector resolution / previous-generation transfer: **${result.comparisons.warmCacheToTransferVectorResolutionRatio.toFixed(3)}×**
- Shared-cache allocated bytes / dense-store allocated bytes: **${result.comparisons.persistentCacheToDenseStoreAllocatedBytesRatio.toFixed(3)}×**
- Shared-cache allocated bytes per dense occurrence: **${formatBytes(result.comparisons.persistentCacheAllocatedBytesPerDenseOccurrence)}**
- Shared-cache allocated bytes per unique embedding input: **${formatBytes(result.comparisons.persistentCacheAllocatedBytesPerUniqueEmbeddingInput)}**

## Validation

| Lane | All canaries | Lexical/source rows | Dense rows | Projection rows after reopen | Physical projections |
| --- | ---: | ---: | ---: | ---: | ---: |
${validationRows}

## Controls

- Embedding requests were strictly sequential with request concurrency 1 and batch size ${result.model.requestBatchSize}.
- Only \`cold_no_cache\` called the served model. Transfer and warm-cache lanes failed on any miss.
- The live ${result.model.nativeDimensions.toLocaleString()}-dimension Octen output was projected to the first ${result.model.storedDimensions.toLocaleString()} dimensions and L2-normalized before storage.
- Scratch source files were evicted with file-specific \`POSIX_FADV_DONTNEED\` before every lane.
- The previous dense store and shared cache were separately evicted before their measured reads.
- Original Pi sessions, the production recall generation, and the production embedding cache were never opened for writing.

## Caveats

${result.caveats.map((caveat) => `- ${caveat}`).join('\n')}

## Measured implication

The persistent cache removed the cold model phase, but it did not improve the normal replacement path. Verified transfer completed the whole replacement in ${formatMilliseconds(result.lanes.find(({ lane }) => lane === 'previous_generation_transfer')?.phases.totalMilliseconds ?? 0)}, while the warm-cache replacement took ${formatMilliseconds(result.lanes.find(({ lane }) => lane === 'warm_shared_cache')?.phases.totalMilliseconds ?? 0)}. Direct vector transfer was ${result.comparisons.warmCacheToTransferVectorResolutionRatio.toFixed(3)}× faster than reading the cold file cache.

The cache added ${formatBytes(result.cacheSeed.cacheSize.allocatedBytes)} and ${result.cacheSeed.cacheSize.files.toLocaleString()} files—${result.comparisons.persistentCacheToDenseStoreAllocatedBytesRatio.toFixed(3)}× the allocated size of the searchable dense store it duplicated.

## Recommendation

Do not retain a persistent embedding cache in the new topology. Use the active or interrupted generation as the verified vector source, deduplicate cold builds against vectors already written to their dense store, and recompute from immutable sessions when no valid vector source survives.

Reconsider only if future evidence shows that cold source-only rebuilds happen often enough for their avoided model time to outweigh a second vector corpus and its lifecycle.
`;
}

async function readGitRevision(): Promise<string> {
  const { stdout } = await execFileAsync('git', ['-C', REPOSITORY_ROOT, 'rev-parse', 'HEAD']);
  return stdout.trim();
}

async function verifyLiveEmbeddingService(
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
  nativeEmbeddingClient: LocalEmbeddingClient,
): Promise<void> {
  const healthUrl = `${config.embeddingBaseUrl.replace(/\/v1\/?$/u, '')}/health`;
  const health = await fetch(healthUrl, { signal: AbortSignal.timeout(5_000) });
  if (!health.ok) {
    throw new Error(`Recall no-cache benchmark embedding health failed: ${health.status}`);
  }
  const warmup = await nativeEmbeddingClient.embedTexts([
    'Issue 112 sequential no-cache benchmark warm-up request.',
  ]);
  const nativeVector = warmup[0];
  if (!nativeVector) {
    throw new Error('Recall no-cache benchmark embedding warm-up returned no vector');
  }
  assertProjectedEmbedding(projectOctenEmbeddingPrefix(nativeVector));
}

async function main(): Promise<void> {
  const arguments_ = parseBenchmarkArguments(process.argv.slice(2));
  const config = await loadRecallConversationConfig();
  const runDirectory = await createSafeScratchRunDirectory({
    scratchParent: arguments_.scratchParent,
    protectedRoots: [
      resolve(config.sessionsDirectory),
      resolve(config.dataDirectory),
      REPOSITORY_ROOT,
    ],
    mode: arguments_.mode,
  });
  let completed = false;
  try {
    const snapshotDirectory = join(runDirectory, 'session-snapshot');
    console.log(
      `[sample] freezing ${arguments_.mode === 'full' ? '10%' : 'pilot'} size-stratified sample`,
    );
    const sample = await createFrozenSizeStratifiedRecallSample({
      sessionsDirectory: config.sessionsDirectory,
      snapshotDirectory,
      sampleRate: FULL_SAMPLE_RATE,
      ...(arguments_.mode === 'pilot' ? { pilotFileCount: PILOT_FILE_COUNT } : {}),
    });
    console.log(
      `[sample] ${sample.summary.selectedFiles}/${sample.summary.populationFiles} files; ` +
        `${(sample.summary.selectedByteRate * 100).toFixed(2)}% of source bytes`,
    );

    const tokenizerCacheDirectory = join(runDirectory, 'tokenizer-cache');
    await cp(config.tokenizerCacheDirectory, tokenizerCacheDirectory, {
      recursive: true,
      force: false,
    });
    const tokenizer = await loadOctenConversationTokenizer({
      cacheDirectory: tokenizerCacheDirectory,
    });
    const nativeEmbeddingClient = createLocalEmbeddingClient({
      baseUrl: config.embeddingBaseUrl,
      model: config.embeddingModel,
      dimensions: NATIVE_OCTEN_DIMENSIONS,
      batchSize: EMBEDDING_REQUEST_BATCH_SIZE,
      requestTimeoutMilliseconds: 15 * 60 * 1_000,
    });
    console.log('[model] verifying live service with one excluded sequential warm-up request');
    await verifyLiveEmbeddingService(config, nativeEmbeddingClient);

    const generationsDirectory = join(runDirectory, 'generations');
    await mkdir(generationsDirectory, { recursive: true, mode: 0o700 });
    const privateProgressPath = join(runDirectory, 'private-progress.json');
    const textOnly = await runBenchmarkLane({
      lane: 'text_only',
      vectorMode: 'none',
      generationDirectory: join(generationsDirectory, 'text-only'),
      generationId: 'prototype-text-only',
      sample,
      tokenizer,
      config,
      nativeEmbeddingClient,
      privateProgressPath,
    });
    const coldNoCacheDirectory = join(generationsDirectory, 'cold-no-cache');
    const coldNoCache = await runBenchmarkLane({
      lane: 'cold_no_cache',
      vectorMode: 'model',
      generationDirectory: coldNoCacheDirectory,
      generationId: 'prototype-cold-no-cache',
      sample,
      tokenizer,
      config,
      nativeEmbeddingClient,
      privateProgressPath,
    });
    const transfer = await runBenchmarkLane({
      lane: 'previous_generation_transfer',
      vectorMode: 'transfer',
      generationDirectory: join(generationsDirectory, 'previous-generation-transfer'),
      generationId: 'prototype-previous-generation-transfer',
      sample,
      tokenizer,
      config,
      nativeEmbeddingClient,
      transferSourceGenerationDirectory: coldNoCacheDirectory,
      privateProgressPath,
    });

    const acceptedStrata = new Set(
      coldNoCache.files.filter(({ accepted }) => accepted).map(({ stratumIndex }) => stratumIndex),
    );
    const cacheDirectory = join(runDirectory, 'embedding-cache');
    const seededCache = await seedScratchEmbeddingCache({
      sample,
      acceptedStrata,
      sourceGenerationDirectory: coldNoCacheDirectory,
      cacheDirectory,
      tokenizer,
      config,
    });
    await evictScratchPageCache([cacheDirectory]);
    let failClosedCacheProviderCalls = 0;
    const warmCache = createEmbeddingVectorCache({
      cacheDirectory,
      identity: createEmbeddingCacheIdentity(config),
      embeddingRequestBatchSize: EMBEDDING_REQUEST_BATCH_SIZE,
      embeddings: {
        async embedTexts() {
          failClosedCacheProviderCalls += 1;
          throw new Error('Recall no-cache benchmark warm cache attempted a model request');
        },
      },
    });
    const warmSharedCache = await runBenchmarkLane({
      lane: 'warm_shared_cache',
      vectorMode: 'cache',
      generationDirectory: join(generationsDirectory, 'warm-shared-cache'),
      generationId: 'prototype-warm-shared-cache',
      sample,
      tokenizer,
      config,
      nativeEmbeddingClient,
      embeddingCache: warmCache,
      privateProgressPath,
    });
    if (failClosedCacheProviderCalls !== 0) {
      throw new Error('Recall no-cache benchmark warm cache provider was invoked');
    }
    const lanes = [textOnly, coldNoCache, transfer, warmSharedCache];
    assertBenchmarkLaneEquivalence(lanes);

    const result: RecallNoCacheBenchmarkResult = {
      prototype: 'recall-no-persistent-embedding-cache-v1',
      mode: arguments_.mode,
      generatedAt: new Date().toISOString(),
      gitRevision: await readGitRevision(),
      model: {
        requestModel: config.embeddingModel,
        servedModelId: config.embeddingServedModelId,
        nativeDimensions: NATIVE_OCTEN_DIMENSIONS,
        storedDimensions: TARGET_EMBEDDING_DIMENSIONS,
        projection: 'first-n-then-l2-v1',
        requestConcurrency: 1,
        requestBatchSize: EMBEDDING_REQUEST_BATCH_SIZE,
      },
      chunkPolicy: CHUNK_POLICY,
      sample: sample.summary,
      sampleSourceFilesExcludedAsActive: 0,
      pageCacheControl: {
        method: 'posix_fadvise-dontneed',
        sourceFilesEvictedBeforeEachLane: true,
        transferStoreEvictedBeforeRead: true,
        sharedCacheEvictedBeforeRead: true,
      },
      lanes,
      cacheSeed: seededCache.result,
      comparisons: {
        coldNoCacheToTextOnlyTotalRatio: safeRatio(
          coldNoCache.phases.totalMilliseconds,
          textOnly.phases.totalMilliseconds,
        ),
        transferToWarmCacheTotalRatio: safeRatio(
          transfer.phases.totalMilliseconds,
          warmSharedCache.phases.totalMilliseconds,
        ),
        transferToWarmCacheVectorResolutionRatio: safeRatio(
          transfer.phases.vectorResolutionMilliseconds,
          warmSharedCache.phases.vectorResolutionMilliseconds,
        ),
        warmCacheToTransferVectorResolutionRatio: safeRatio(
          warmSharedCache.phases.vectorResolutionMilliseconds,
          transfer.phases.vectorResolutionMilliseconds,
        ),
        persistentCacheAllocatedBytesPerDenseOccurrence: safeRatio(
          seededCache.result.cacheSize.allocatedBytes,
          coldNoCache.counts.denseEvidenceOccurrences,
        ),
        persistentCacheAllocatedBytesPerUniqueEmbeddingInput: safeRatio(
          seededCache.result.cacheSize.allocatedBytes,
          coldNoCache.counts.modelEmbeddedUniqueInputs,
        ),
        persistentCacheToDenseStoreAllocatedBytesRatio: safeRatio(
          seededCache.result.cacheSize.allocatedBytes,
          coldNoCache.sizesAfterOptimize.dense.allocatedBytes,
        ),
      },
      caveats: [
        arguments_.mode === 'full'
          ? 'The sample is one deterministic 10% size-stratified draw, not repeated random samples.'
          : 'The pilot is a size-distributed subset of the eventual 10% sample and is not decision evidence by itself.',
        'The production projection builder currently performs a second import/tokenization pass; projection preparation reports that cost separately.',
        'POSIX_FADV_DONTNEED is an advisory, file-scoped cache control and does not evict directory metadata.',
        'Only one cold model pass was run; model-service load and co-resident GPU activity may affect its absolute time.',
        'The split lexical schema is based on the accepted prototype plus current provenance fields, but issue 112 has not frozen every final scalar column.',
        'All sources are treated as eligible for this upper-bound rebuild comparison; the future recall horizon may reduce row counts.',
      ],
    };
    const reportPath = join(
      BENCHMARK_DIRECTORY,
      arguments_.mode === 'pilot' ? 'PILOT-MEASUREMENTS.md' : 'MEASUREMENTS.md',
    );
    const jsonPath = join(
      BENCHMARK_DIRECTORY,
      arguments_.mode === 'pilot' ? 'pilot-results.json' : 'benchmark-results.json',
    );
    await writeFile(reportPath, formatBenchmarkReport(result), 'utf8');
    await writeFile(jsonPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
    console.log(`\n[result] wrote ${relative(REPOSITORY_ROOT, reportPath)}`);
    completed = true;
  } finally {
    if (completed && !arguments_.keepScratch) {
      console.log('[cleanup] removing private scratch snapshot, stores, and cache');
      await removeScratchRunDirectory(runDirectory);
    } else {
      console.log(`[scratch] retained private run directory: ${runDirectory}`);
    }
  }
}

await main();
