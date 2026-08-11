// PROTOTYPE ONLY: validates one bounded Octen-Embedding-0.6B local-inference candidate.
import { createHash } from 'node:crypto';
import { readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { cpus } from 'node:os';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';

import { getLlama, LlamaLogLevel } from 'node-llama-cpp';

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
  'local-octen-0.6b-prototype',
);
const MODEL_PATH = join(SCRATCH_DIRECTORY, 'models', 'Octen-Embedding-0.6B-Q8_0.gguf');
const DATABASE_PATH = join(SCRATCH_DIRECTORY, 'sample-recall.sqlite');
const PRIVATE_SAMPLE_MANIFEST_PATH = join(SCRATCH_DIRECTORY, 'selected-sessions.json');
const RESULTS_PATH = join(SCRATCH_DIRECTORY, 'prototype-results.json');
const QUALITY_WORK_DIRECTORY = join(
  PROJECT_DIRECTORY,
  'evaluation',
  '.recall-data',
  'recall-quality-evaluation',
);
const MODEL_SHA256 = '2dbae3888c66d4ef39adb09294b54e14dae7d7246c36620d9a8d9ec8502b5542';
const MODEL_BYTES = 639_150_720;
const MINIMUM_SESSION_COUNT = 100;
const MAXIMUM_SESSION_COUNT = 200;
const MINIMUM_DENSE_DOCUMENTS = 10_000;
const MAXIMUM_DENSE_DOCUMENTS = 20_000;
const MAXIMUM_RUNTIME_MILLISECONDS = 2 * 60 * 60 * 1_000;
const MAXIMUM_SCRATCH_BYTES = 8 * 1024 ** 3;
const MINIMUM_FREE_BYTES = 240 * 1024 ** 3;
const EMBEDDING_PARALLELISM = 4;
const EMBEDDING_THREADS_PER_CONTEXT = Math.max(
  1,
  Math.floor(cpus().length / EMBEDDING_PARALLELISM),
);
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
    throw new Error('Local Octen prototype could not read available storage');
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

async function assertPrototypeSafety(): Promise<void> {
  if (!HOME_DIRECTORY || !SCRATCH_DIRECTORY.includes('/.pi/agent/recall-debug/')) {
    throw new Error(`Local Octen prototype scratch path is unsafe: ${SCRATCH_DIRECTORY}`);
  }
  const freeBytes = await readAvailableBytes(SCRATCH_DIRECTORY);
  if (freeBytes < MINIMUM_FREE_BYTES) {
    throw new Error(
      `Local Octen prototype requires ${MINIMUM_FREE_BYTES} free bytes; found ${freeBytes}`,
    );
  }
  const modelStat = await stat(MODEL_PATH);
  if (modelStat.size !== MODEL_BYTES) {
    throw new Error(
      `Local Octen prototype model size mismatch: expected ${MODEL_BYTES}, received ${modelStat.size}`,
    );
  }
  const modelHash = createHash('sha256')
    .update(await readFile(MODEL_PATH))
    .digest('hex');
  if (modelHash !== MODEL_SHA256) {
    throw new Error(
      `Local Octen prototype model checksum mismatch: expected ${MODEL_SHA256}, received ${modelHash}`,
    );
  }
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
      `Local Octen prototype sample too small: selected ${selected.length} sessions and ${selectedDenseDocuments} Dense documents`,
    );
  }
  return { selected, parseFailures, discovered: discoveredPaths.length };
}

async function createPrototypeEmbeddingProvider(): Promise<
  RecallEmbeddingProvider & { dispose(): Promise<void>; readMetrics(): object }
> {
  const loadStartedAt = performance.now();
  const llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
  const model = await llama.loadModel({
    modelPath: MODEL_PATH,
    gpuLayers: 0,
    useMmap: true,
    checkTensors: true,
  });
  if (model.embeddingVectorSize !== 1_024) {
    throw new Error(
      `Local Octen prototype vector width mismatch: expected 1024, received ${model.embeddingVectorSize}`,
    );
  }
  const contexts = await Promise.all(
    Array.from({ length: EMBEDDING_PARALLELISM }, () =>
      model.createEmbeddingContext({
        contextSize: 2_048,
        threads: EMBEDDING_THREADS_PER_CONTEXT,
        batchSize: 512,
      }),
    ),
  );
  const loadMilliseconds = performance.now() - loadStartedAt;
  const embeddingLatenciesMilliseconds: number[] = [];
  let embeddedTexts = 0;

  async function embedWithContext(contextIndex: number, text: string): Promise<number[]> {
    const startedAt = performance.now();
    const result = await contexts[contextIndex]!.getEmbeddingFor(text);
    embeddingLatenciesMilliseconds.push(performance.now() - startedAt);
    embeddedTexts += 1;
    if (result.vector.length !== 1_024) {
      throw new Error(
        `Local Octen prototype vector width mismatch: expected 1024, received ${result.vector.length}`,
      );
    }
    const norm = Math.sqrt(result.vector.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm === 0) {
      throw new Error('Local Octen prototype produced a non-finite or zero embedding norm');
    }
    return Array.from(result.vector, (value) => value / norm);
  }

  async function embedTexts(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    const vectors = new Array<number[]>(texts.length);
    let nextIndex = 0;
    await Promise.all(
      contexts.map(async (_, contextIndex) => {
        while (true) {
          if (signal?.aborted) {
            throw new Error('Local Octen prototype embedding cancelled', { cause: signal.reason });
          }
          const index = nextIndex;
          nextIndex += 1;
          const text = texts[index];
          if (text === undefined) {
            return;
          }
          vectors[index] = await embedWithContext(contextIndex, text);
        }
      }),
    );
    return vectors;
  }

  return {
    async embedQuery(query, signal) {
      return (await embedTexts([query], signal))[0]!;
    },
    embedDocuments: embedTexts,
    async dispose() {
      await Promise.all(contexts.map((context) => context.dispose()));
      await model.dispose();
      await llama.dispose();
    },
    readMetrics() {
      return {
        loadMilliseconds,
        embeddedTexts,
        latencyMedianMilliseconds: percentile(embeddingLatenciesMilliseconds, 0.5),
        latencyP95Milliseconds: percentile(embeddingLatenciesMilliseconds, 0.95),
        parallelism: EMBEDDING_PARALLELISM,
        threadsPerContext: EMBEDDING_THREADS_PER_CONTEXT,
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
  const sample = await selectRepresentativeSessions(tokenizer);
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
      embeddingBaseUrl: 'embedded://octen-embedding-0.6b-q8',
      embeddingModel: 'octen-embedding-0.6b-q8',
      embeddingServedModelId: 'Octen/Octen-Embedding-0.6B',
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
        'Local Octen prototype quality evaluation returned no final-five measurement',
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
        `Local Octen prototype exceeded scratch allocation: ${scratchBytes} > ${MAXIMUM_SCRATCH_BYTES}`,
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
        upstreamRepository: 'Octen/Octen-Embedding-0.6B',
        upstreamRevision: 'd715b32ee68f057b54dff09fc93c23485bc403d3',
        prototypeRepository: 'mykor/Octen-Embedding-0.6B-GGUF',
        prototypeRevision: '26f558910bcf99e75dcf4fbed40615a3145a46ee',
        fileName: basename(MODEL_PATH),
        bytes: MODEL_BYTES,
        sha256: MODEL_SHA256,
        quantization: 'Q8_0',
        license: 'Apache-2.0',
      },
      runtime: {
        package: 'node-llama-cpp',
        version: '3.19.1',
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
