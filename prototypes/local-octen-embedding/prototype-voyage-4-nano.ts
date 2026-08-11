// PROTOTYPE ONLY: validates one bounded Voyage 4 Nano local-inference candidate.
import { createHash } from 'node:crypto';
import { readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { cpus } from 'node:os';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';

import { Tokenizer } from '@huggingface/tokenizers';
import ort from 'onnxruntime-node';

import { DEFAULT_RECALL_CHUNK_POLICY } from '../../src/recall-index-manifest.js';
import { indexChangedConversationSessions } from '../../src/incremental-session-indexer.js';
import { listRecallSessionFiles } from '../../src/listRecallSessionFiles.js';
import { loadOctenConversationTokenizer } from '../../src/octen-conversation-tokenizer.js';
import { listIgnoredPhysicalSessionPaths } from '../../src/physical-session-ignore.js';
import type { RecallEmbeddingProvider } from '../../src/recall-inference-capabilities.js';
import { loadRecallQualityCorpus } from '../../src/recall-quality-corpus.js';
import {
  normalizeRecallProjectLineages,
  resolveProjectIdentity,
} from '../../src/resolve-project-identity.js';
import { runRecallQualityEvaluation } from '../../src/run-recall-quality-evaluation.js';
import { readSessionConversationImport } from '../../src/session-conversation-index.js';
import { openSqliteRecallDatabase } from '../../src/sqlite-recall-database.js';

const PROJECT_DIRECTORY = resolve(import.meta.dirname, '..', '..');
const HOME_DIRECTORY = process.env.HOME ?? '';
const SESSION_DIRECTORY = join(HOME_DIRECTORY, '.pi', 'agent', 'sessions');
const RECALL_DIRECTORY = join(HOME_DIRECTORY, '.pi', 'agent', 'recall');
const SCRATCH_DIRECTORY = join(
  HOME_DIRECTORY,
  '.pi',
  'agent',
  'recall-debug',
  'voyage-4-nano-prototype',
);
const OCTEN_SAMPLE_MANIFEST_PATH = join(
  HOME_DIRECTORY,
  '.pi',
  'agent',
  'recall-debug',
  'local-octen-0.6b-prototype',
  'selected-sessions.json',
);
const MODEL_DIRECTORY = join(SCRATCH_DIRECTORY, 'model');
const MODEL_GRAPH_PATH = join(MODEL_DIRECTORY, 'model_quantized.onnx');
const MODEL_WEIGHTS_PATH = join(MODEL_DIRECTORY, 'model_quantized.onnx_data');
const TOKENIZER_PATH = join(MODEL_DIRECTORY, 'tokenizer.json');
const TOKENIZER_CONFIG_PATH = join(MODEL_DIRECTORY, 'tokenizer_config.json');
const DATABASE_PATH = join(SCRATCH_DIRECTORY, 'sample-recall.sqlite');
const PRIVATE_SAMPLE_MANIFEST_PATH = join(SCRATCH_DIRECTORY, 'selected-sessions.json');
const RESULTS_PATH = join(SCRATCH_DIRECTORY, 'prototype-results.json');
const QUALITY_WORK_DIRECTORY = join(
  PROJECT_DIRECTORY,
  'evaluation',
  '.recall-data',
  'recall-quality-evaluation',
);
const MODEL_GRAPH_SHA256 = 'e2999ae63b575e34b7bef6a513f48c7f7701cebfb1bfffcb9db23ab0db324157';
const MODEL_GRAPH_BYTES = 268_076;
const MODEL_WEIGHTS_SHA256 = 'db94fc1655db1a1b55f67f7019ebbd803142f54e1bd83cfd1c2d6faadd6a648c';
const MODEL_WEIGHTS_BYTES = 421_625_856;
const VOYAGE_QUERY_PREFIX = 'Represent the query for retrieving supporting documents: ';
const VOYAGE_DOCUMENT_PREFIX = 'Represent the document for retrieval: ';
const EMBEDDING_BATCH_SIZE = 8;
const MINIMUM_SESSION_COUNT = 100;
const MAXIMUM_SESSION_COUNT = 200;
const MINIMUM_DENSE_DOCUMENTS = 10_000;
const MAXIMUM_DENSE_DOCUMENTS = 20_000;
const MAXIMUM_RUNTIME_MILLISECONDS = 2 * 60 * 60 * 1_000;
const MAXIMUM_SCRATCH_BYTES = 8 * 1024 ** 3;
const MINIMUM_FREE_BYTES = 240 * 1024 ** 3;
const SAMPLE_QUERIES = [
  'SQLite recall database',
  'embedding model setup',
  'systemd index timer',
  'worktree implementation branch',
  'source postings BitPacked',
  'session JSONL parsing',
] as const;

interface SampleSession {
  path: string;
  relativePath: string;
  projectDirectory: string;
  denseDocuments: number;
  invocations: number;
  bytes: number;
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function percentile(values: readonly number[], probability: number): number {
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(probability * ordered.length) - 1);
  return ordered[index] ?? 0;
}

async function readAvailableBytes(path: string): Promise<number> {
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const { stdout } = await promisify(execFile)('df', ['--output=avail', '-B1', path]);
  const value = Number(stdout.trim().split(/\s+/u).at(-1));
  if (!Number.isSafeInteger(value)) {
    throw new Error('Voyage 4 Nano prototype could not read available storage');
  }
  return value;
}

async function readAllocatedBytes(path: string): Promise<number> {
  const pathStat = await stat(path);
  if (!pathStat.isDirectory()) {
    return pathStat.blocks * 512;
  }
  let bytes = pathStat.blocks * 512;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    bytes += await readAllocatedBytes(join(path, entry.name));
  }
  return bytes;
}

async function readDeviceWrittenBytes(): Promise<number | null> {
  const blockDevice = process.env.PI_RECALL_PROTOTYPE_BLOCK_DEVICE;
  if (!blockDevice) {
    return null;
  }
  const fields = (await readFile(`/sys/class/block/${blockDevice}/stat`, 'utf8'))
    .trim()
    .split(/\s+/u);
  const sectorsWritten = Number(fields[6]);
  return Number.isSafeInteger(sectorsWritten) ? sectorsWritten * 512 : null;
}

async function assertChecksummedArtifact(
  path: string,
  expectedBytes: number,
  expectedSha256: string,
): Promise<void> {
  const artifactStat = await stat(path);
  if (artifactStat.size !== expectedBytes) {
    throw new Error(
      `Voyage 4 Nano prototype artifact size mismatch for ${basename(path)}: expected ${expectedBytes}, received ${artifactStat.size}`,
    );
  }
  const artifactHash = createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
  if (artifactHash !== expectedSha256) {
    throw new Error(
      `Voyage 4 Nano prototype artifact checksum mismatch for ${basename(path)}: expected ${expectedSha256}, received ${artifactHash}`,
    );
  }
}

async function assertPrototypeSafety(): Promise<void> {
  if (!HOME_DIRECTORY || !SCRATCH_DIRECTORY.includes('/.pi/agent/recall-debug/')) {
    throw new Error(`Voyage 4 Nano prototype scratch path is unsafe: ${SCRATCH_DIRECTORY}`);
  }
  const freeBytes = await readAvailableBytes(SCRATCH_DIRECTORY);
  if (freeBytes < MINIMUM_FREE_BYTES) {
    throw new Error(
      `Voyage 4 Nano prototype requires ${MINIMUM_FREE_BYTES} free bytes; found ${freeBytes}`,
    );
  }
  await assertChecksummedArtifact(MODEL_GRAPH_PATH, MODEL_GRAPH_BYTES, MODEL_GRAPH_SHA256);
  await assertChecksummedArtifact(MODEL_WEIGHTS_PATH, MODEL_WEIGHTS_BYTES, MODEL_WEIGHTS_SHA256);
}

async function selectRepresentativeSessions(
  tokenizer: Awaited<ReturnType<typeof loadOctenConversationTokenizer>>,
): Promise<{ selected: SampleSession[]; parseFailures: number; discovered: number }> {
  const ignored = new Set(
    await listIgnoredPhysicalSessionPaths(join(RECALL_DIRECTORY, 'physical-session-ignore.json')),
  );
  const discoveredPaths = (await listRecallSessionFiles(SESSION_DIRECTORY)).filter(
    (sessionPath) => !ignored.has(sessionPath),
  );
  const byProjectDirectory = new Map<string, string[]>();
  for (const sessionPath of discoveredPaths) {
    const relativePath = relative(SESSION_DIRECTORY, sessionPath);
    const projectDirectory = relativePath.split(sep)[0] ?? 'unknown';
    const paths = byProjectDirectory.get(projectDirectory) ?? [];
    paths.push(sessionPath);
    byProjectDirectory.set(projectDirectory, paths);
  }
  for (const paths of byProjectDirectory.values()) {
    paths.sort((left, right) =>
      sha256Text(relative(SESSION_DIRECTORY, left)).localeCompare(
        sha256Text(relative(SESSION_DIRECTORY, right)),
      ),
    );
  }
  const projectDirectories = [...byProjectDirectory.keys()].sort((left, right) =>
    sha256Text(left).localeCompare(sha256Text(right)),
  );
  const candidates: string[] = [];
  for (let index = 0; ; index += 1) {
    let added = false;
    for (const projectDirectory of projectDirectories) {
      const path = byProjectDirectory.get(projectDirectory)?.[index];
      if (path) {
        candidates.push(path);
        added = true;
      }
    }
    if (!added) {
      break;
    }
  }

  const selected: SampleSession[] = [];
  let selectedDenseDocuments = 0;
  let parseFailures = 0;
  for (const sessionPath of candidates) {
    if (
      selected.length >= MINIMUM_SESSION_COUNT &&
      selectedDenseDocuments >= MINIMUM_DENSE_DOCUMENTS
    ) {
      break;
    }
    if (selected.length >= MAXIMUM_SESSION_COUNT) {
      break;
    }
    let imported: Awaited<ReturnType<typeof readSessionConversationImport>>;
    try {
      imported = await readSessionConversationImport(sessionPath, {
        tokenizer,
        ...DEFAULT_RECALL_CHUNK_POLICY,
      });
    } catch {
      parseFailures += 1;
      continue;
    }
    const denseDocuments = imported.chunks.filter((chunk) => chunk.isDenseSearchable).length;
    if (selectedDenseDocuments + denseDocuments > MAXIMUM_DENSE_DOCUMENTS) {
      continue;
    }
    const fileStat = await stat(sessionPath);
    const relativePath = relative(SESSION_DIRECTORY, sessionPath);
    selected.push({
      path: sessionPath,
      relativePath,
      projectDirectory: relativePath.split(sep)[0] ?? 'unknown',
      denseDocuments,
      invocations: imported.invocations.length,
      bytes: fileStat.size,
    });
    selectedDenseDocuments += denseDocuments;
  }
  if (selected.length < MINIMUM_SESSION_COUNT || selectedDenseDocuments < MINIMUM_DENSE_DOCUMENTS) {
    throw new Error(
      `Voyage 4 Nano prototype sample too small: selected ${selected.length} sessions and ${selectedDenseDocuments} Dense documents`,
    );
  }
  return { selected, parseFailures, discovered: discoveredPaths.length };
}

async function loadExactOctenSample(): Promise<{
  selected: SampleSession[];
  parseFailures: number;
  discovered: number;
}> {
  const raw: unknown = JSON.parse(await readFile(OCTEN_SAMPLE_MANIFEST_PATH, 'utf8'));
  if (typeof raw !== 'object' || raw === null || !Array.isArray(Reflect.get(raw, 'sessions'))) {
    throw new Error('Voyage 4 Nano prototype requires the exact Octen sample manifest');
  }
  const selected: SampleSession[] = [];
  for (const value of Reflect.get(raw, 'sessions') as unknown[]) {
    if (typeof value !== 'object' || value === null) {
      throw new Error('Voyage 4 Nano prototype sample manifest contains an invalid session');
    }
    const path = Reflect.get(value, 'path');
    const relativePath = Reflect.get(value, 'relativePath');
    const denseDocuments = Reflect.get(value, 'denseDocuments');
    const invocations = Reflect.get(value, 'invocations');
    const bytes = Reflect.get(value, 'bytes');
    if (
      typeof path !== 'string' ||
      typeof relativePath !== 'string' ||
      typeof denseDocuments !== 'number' ||
      typeof invocations !== 'number' ||
      typeof bytes !== 'number' ||
      resolve(SESSION_DIRECTORY, relativePath) !== path ||
      (await stat(path)).size !== bytes
    ) {
      throw new Error('Voyage 4 Nano prototype sample drifted after the Octen run');
    }
    selected.push({
      path,
      relativePath,
      projectDirectory: relativePath.split(sep)[0] ?? 'unknown',
      denseDocuments,
      invocations,
      bytes,
    });
  }
  if (
    selected.length !== 171 ||
    selected.reduce((sum, session) => sum + session.denseDocuments, 0) !== 12_966
  ) {
    throw new Error('Voyage 4 Nano prototype sample does not match the bounded Octen corpus');
  }
  return { selected, parseFailures: 0, discovered: 3_737 };
}

async function createPrototypeEmbeddingProvider(): Promise<
  RecallEmbeddingProvider & { dispose(): Promise<void>; readMetrics(): object }
> {
  const tokenizer = new Tokenizer(
    JSON.parse(await readFile(TOKENIZER_PATH, 'utf8')) as object,
    JSON.parse(await readFile(TOKENIZER_CONFIG_PATH, 'utf8')) as object,
  );
  const loadStartedAt = performance.now();
  const session = await ort.InferenceSession.create(MODEL_GRAPH_PATH, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    executionMode: 'sequential',
    intraOpNumThreads: 8,
    interOpNumThreads: 1,
  });
  const loadMilliseconds = performance.now() - loadStartedAt;
  if (!session.outputNames.includes('pooler_output')) {
    throw new Error('Voyage 4 Nano prototype ONNX graph is missing pooler_output');
  }
  const batchLatenciesMilliseconds: number[] = [];
  let embeddedTexts = 0;
  let embeddedTokens = 0;

  function encodeInput(text: string, prefix: string): number[] {
    return [
      ...tokenizer.encode(`${prefix}${text}`, {
        add_special_tokens: true,
        return_token_type_ids: false,
      }).ids,
    ];
  }

  async function embedBatch(
    batch: readonly { index: number; ids: readonly number[] }[],
    vectors: number[][],
  ): Promise<void> {
    const width = Math.max(...batch.map((item) => item.ids.length));
    const inputIds = new BigInt64Array(batch.length * width).fill(151_643n);
    const attentionMask = new BigInt64Array(batch.length * width);
    for (const [row, item] of batch.entries()) {
      for (const [column, token] of item.ids.entries()) {
        inputIds[row * width + column] = BigInt(token);
        attentionMask[row * width + column] = 1n;
      }
    }
    const startedAt = performance.now();
    const outputs = await session.run({
      input_ids: new ort.Tensor('int64', inputIds, [batch.length, width]),
      attention_mask: new ort.Tensor('int64', attentionMask, [batch.length, width]),
    });
    batchLatenciesMilliseconds.push(performance.now() - startedAt);
    const output = outputs.pooler_output;
    if (!output || output.dims[0] !== batch.length || output.dims[1] !== 2_048) {
      throw new Error('Voyage 4 Nano prototype ONNX output shape is not [batch, 2048]');
    }
    const values = output.data as Float32Array;
    for (const [row, item] of batch.entries()) {
      const offset = row * 2_048;
      let squaredNorm = 0;
      for (let dimension = 0; dimension < 1_024; dimension += 1) {
        const value = values[offset + dimension]!;
        squaredNorm += value * value;
      }
      const norm = Math.sqrt(squaredNorm);
      if (!Number.isFinite(norm) || norm === 0) {
        throw new Error('Voyage 4 Nano prototype produced a non-finite or zero embedding norm');
      }
      vectors[item.index] = Array.from(
        { length: 1_024 },
        (_, dimension) => values[offset + dimension]! / norm,
      );
    }
  }

  async function embedTexts(
    texts: readonly string[],
    prefix: string,
    signal?: AbortSignal,
  ): Promise<number[][]> {
    const encoded = texts
      .map((text, index) => ({ index, ids: encodeInput(text, prefix) }))
      .sort((left, right) => left.ids.length - right.ids.length);
    const vectors = new Array<number[]>(texts.length);
    for (let start = 0; start < encoded.length; start += EMBEDDING_BATCH_SIZE) {
      if (signal?.aborted) {
        throw new Error('Voyage 4 Nano prototype embedding cancelled', {
          cause: signal.reason,
        });
      }
      const batch = encoded.slice(start, start + EMBEDDING_BATCH_SIZE);
      embeddedTexts += batch.length;
      embeddedTokens += batch.reduce((sum, item) => sum + item.ids.length, 0);
      await embedBatch(batch, vectors);
    }
    return vectors;
  }

  return {
    async embedQuery(query, signal) {
      return (await embedTexts([query], VOYAGE_QUERY_PREFIX, signal))[0]!;
    },
    async embedDocuments(documents, signal) {
      return embedTexts(documents, VOYAGE_DOCUMENT_PREFIX, signal);
    },
    async dispose() {
      await session.release();
    },
    readMetrics() {
      return {
        loadMilliseconds,
        embeddedTexts,
        embeddedTokens,
        batchLatencyMedianMilliseconds: percentile(batchLatenciesMilliseconds, 0.5),
        batchLatencyP95Milliseconds: percentile(batchLatenciesMilliseconds, 0.95),
        batchSize: EMBEDDING_BATCH_SIZE,
        intraOpThreads: 8,
      };
    },
  };
}

async function main(): Promise<void> {
  await assertPrototypeSafety();
  const startedAt = performance.now();
  const deadline = startedAt + MAXIMUM_RUNTIME_MILLISECONDS;
  let peakRssBytes = process.memoryUsage().rss;
  const memoryPoll = setInterval(() => {
    peakRssBytes = Math.max(peakRssBytes, process.memoryUsage().rss);
    if (performance.now() > deadline) {
      process.kill(process.pid, 'SIGTERM');
    }
  }, 100);
  const tokenizer = await loadOctenConversationTokenizer({
    cacheDirectory: join(SCRATCH_DIRECTORY, 'tokenizers'),
  });
  const sample = await loadExactOctenSample();
  await writeFile(
    PRIVATE_SAMPLE_MANIFEST_PATH,
    `${JSON.stringify(
      {
        selectedAt: new Date().toISOString(),
        sessions: sample.selected.map((session) => ({
          path: session.path,
          relativePath: session.relativePath,
          denseDocuments: session.denseDocuments,
          invocations: session.invocations,
          bytes: session.bytes,
        })),
      },
      null,
      2,
    )}\n`,
    { mode: 0o600 },
  );
  const sampleHash = sha256Text(sample.selected.map((session) => session.relativePath).join('\n'));
  const provider = await createPrototypeEmbeddingProvider();
  try {
    const qualityCorpus = await loadRecallQualityCorpus(
      join(PROJECT_DIRECTORY, 'evaluation', 'recall-quality-cases.json'),
    );
    const scratchConfig = {
      sessionsDirectory: qualityCorpus.sessionDirectory,
      sqliteDatabasePath: join(SCRATCH_DIRECTORY, 'quality-recall.sqlite'),
      manifestPath: join(SCRATCH_DIRECTORY, 'quality-index-manifest.json'),
      indexMaintenanceStatusPath: join(SCRATCH_DIRECTORY, 'quality-maintenance-status.json'),
      physicalSessionIgnoreStatePath: join(SCRATCH_DIRECTORY, 'quality-ignore.json'),
      tokenizerCacheDirectory: join(SCRATCH_DIRECTORY, 'tokenizers'),
      lockPath: join(SCRATCH_DIRECTORY, 'quality-operation.lock'),
      databaseGenerationRootPath: join(SCRATCH_DIRECTORY, 'quality-generations'),
      embeddingBaseUrl: 'embedded://voyage-4-nano-onnx-int8',
      embeddingModel: 'voyage-4-nano-onnx-int8',
      embeddingServedModelId: 'voyageai/voyage-4-nano',
      embeddingNativeDimensions: 1_024,
      embeddingStoredDimensions: 1_024,
      embeddingBatchSize: 128,
      projectLineages: normalizeRecallProjectLineages({}),
      searchCandidateLimits: { dense: 8, invocation: 8 },
      chunkPolicy: { ...DEFAULT_RECALL_CHUNK_POLICY },
    };
    const quality = await runRecallQualityEvaluation({
      corpus: qualityCorpus,
      baseConfig: scratchConfig,
      workDirectory: QUALITY_WORK_DIRECTORY,
      dependencies: {
        embeddingProvider: provider,
        loadTokenizer: async () => tokenizer,
      },
    });
    const qualityConfiguration = quality.configurations[0];
    const qualityMeasurement = qualityConfiguration?.measurement;
    const qualityFinalMeasurement = qualityMeasurement?.finalCounts.find(
      (measurement) => measurement.finalCount === 5,
    );
    if (!qualityMeasurement || !qualityFinalMeasurement) {
      throw new Error(
        'Voyage 4 Nano prototype quality evaluation returned no final-five measurement',
      );
    }

    await rm(DATABASE_PATH, { force: true });
    await rm(`${DATABASE_PATH}-shm`, { force: true });
    await rm(`${DATABASE_PATH}-wal`, { force: true });
    const writesBefore = await readDeviceWrittenBytes();
    const indexStartedAt = performance.now();
    const indexSummary = await indexChangedConversationSessions({
      sessionsDirectory: SESSION_DIRECTORY,
      selectedPhysicalSessionPaths: sample.selected.map((session) => session.path),
      databasePath: DATABASE_PATH,
      embeddingProvider: provider,
      tokenizer,
      chunkPolicy: DEFAULT_RECALL_CHUNK_POLICY,
      ignoredPhysicalSessionPaths: new Set(),
      resolveProjectIdentity,
      rebuild: true,
      onProgress(event) {
        if (event.kind === 'indexing-maintenance-workset' && event.completedFiles % 10 === 0) {
          process.stderr.write(
            `sample ${event.completedFiles}/${event.totalFiles} · ${event.newlyEmbeddedDocuments} embeddings\n`,
          );
        }
      },
    });
    const indexDurationMilliseconds = performance.now() - indexStartedAt;
    const writesAfter = await readDeviceWrittenBytes();
    const database = openSqliteRecallDatabase(DATABASE_PATH, { readOnly: true });
    const queryMeasurements = [];
    try {
      for (const query of SAMPLE_QUERIES) {
        const queryStartedAt = performance.now();
        const embedding = await provider.embedQuery(query);
        const snapshot = await database.searchRecallSnapshot({
          query,
          embedding,
          denseLimit: 8,
          invocationLimit: 8,
        });
        queryMeasurements.push({
          queryId: sha256Text(query).slice(0, 12),
          latencyMilliseconds: performance.now() - queryStartedAt,
          denseResults: snapshot.denseCandidates.length,
          invocationResults: snapshot.invocationCandidates.length,
        });
      }
    } finally {
      database.close();
    }
    const scratchBytes = await readAllocatedBytes(SCRATCH_DIRECTORY);
    if (scratchBytes > MAXIMUM_SCRATCH_BYTES) {
      throw new Error(
        `Voyage 4 Nano prototype exceeded scratch allocation: ${scratchBytes} > ${MAXIMUM_SCRATCH_BYTES}`,
      );
    }
    const practicalQualityPassed = [
      qualityMeasurement.candidatePoolRecall,
      qualityFinalMeasurement.finalRecall,
      qualityFinalMeasurement.contextUsefulness,
      qualityFinalMeasurement.sourceOccurrencePreservation,
    ].every((measurement) => measurement >= 0.9);
    const result = {
      version: 1,
      completedAt: new Date().toISOString(),
      artifact: {
        upstreamRepository: 'voyageai/voyage-4-nano',
        upstreamRevision: '67fabc9bef010dabc5f6024aa1b1b6b93410426f',
        prototypeRepository: 'onnx-community/voyage-4-nano-ONNX',
        prototypeRevision: 'fb6f32cceb63b6eac92e1fdc08e00d03c41cdf5a',
        files: [
          {
            fileName: basename(MODEL_GRAPH_PATH),
            bytes: MODEL_GRAPH_BYTES,
            sha256: MODEL_GRAPH_SHA256,
          },
          {
            fileName: basename(MODEL_WEIGHTS_PATH),
            bytes: MODEL_WEIGHTS_BYTES,
            sha256: MODEL_WEIGHTS_SHA256,
          },
        ],
        quantization: 'INT8 dynamic',
        nativeDimensions: 2_048,
        storedMrlDimensions: 1_024,
        license: 'Apache-2.0',
      },
      runtime: {
        package: 'onnxruntime-node',
        version: '1.27.0',
        platform: process.platform,
        architecture: process.arch,
        node: process.version,
        cpu: cpus()[0]?.model ?? 'unknown',
        logicalCpus: cpus().length,
        peakRssBytes,
        ...provider.readMetrics(),
      },
      sample: {
        selectionAlgorithm: 'project-round-robin-sha256-v1',
        selectionHash: sampleHash,
        discoveredSessions: sample.discovered,
        selectedSessions: sample.selected.length,
        representedProjectDirectories: new Set(
          sample.selected.map((session) => session.projectDirectory),
        ).size,
        sourceBytes: sample.selected.reduce((sum, session) => sum + session.bytes, 0),
        projectedDenseDocuments: sample.selected.reduce(
          (sum, session) => sum + session.denseDocuments,
          0,
        ),
        projectedInvocations: sample.selected.reduce(
          (sum, session) => sum + session.invocations,
          0,
        ),
        parseFailuresDuringSelection: sample.parseFailures,
      },
      quality: {
        frozenGatePassed: quality.selection.passed,
        practicalLocalGatePassed: practicalQualityPassed,
        practicalMinimum: 0.9,
        candidatePoolRecall: qualityMeasurement.candidatePoolRecall,
        finalRecall: qualityFinalMeasurement.finalRecall,
        contextUsefulness: qualityFinalMeasurement.contextUsefulness,
        sourceOccurrencePreservation: qualityFinalMeasurement.sourceOccurrencePreservation,
        queryLatency: qualityMeasurement.queryLatencyMilliseconds,
        strictGateBlockers: quality.selection.blockers,
      },
      sampleIndex: {
        durationMilliseconds: indexDurationMilliseconds,
        denseDocumentsPerSecond:
          indexSummary.newlyEmbeddedChunks / (indexDurationMilliseconds / 1_000),
        summary: indexSummary,
        deviceWrittenBytes:
          writesBefore === null || writesAfter === null ? null : writesAfter - writesBefore,
        scratchAllocatedBytes: scratchBytes,
      },
      sampleQueries: {
        measurements: queryMeasurements,
        latencyMedianMilliseconds: percentile(
          queryMeasurements.map((measurement) => measurement.latencyMilliseconds),
          0.5,
        ),
        latencyP95Milliseconds: percentile(
          queryMeasurements.map((measurement) => measurement.latencyMilliseconds),
          0.95,
        ),
      },
      bounds: {
        maximumSessions: MAXIMUM_SESSION_COUNT,
        maximumDenseDocuments: MAXIMUM_DENSE_DOCUMENTS,
        maximumRuntimeMilliseconds: MAXIMUM_RUNTIME_MILLISECONDS,
        maximumScratchBytes: MAXIMUM_SCRATCH_BYTES,
        minimumFreeBytes: MINIMUM_FREE_BYTES,
        productionDatabaseMutated: false,
      },
      passed:
        practicalQualityPassed &&
        indexSummary.failedSessions.length === 0 &&
        peakRssBytes <= 4 * 1024 ** 3 &&
        queryMeasurements.every((measurement) => measurement.latencyMilliseconds < 500),
    };
    await writeFile(RESULTS_PATH, `${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (!result.passed) {
      process.exitCode = 2;
    }
  } finally {
    clearInterval(memoryPoll);
    await provider.dispose();
    await rm(QUALITY_WORK_DIRECTORY, { recursive: true, force: true });
  }
}

await main();
