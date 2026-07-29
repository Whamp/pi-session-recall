import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { chmod, constants, copyFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve, sep } from 'node:path';

import { listRecallConversationSessionFiles } from '../recall-conversation-corpus.js';

/** Fixed sampling seed that makes issue 112 benchmark membership reproducible. */
export const RECALL_NO_CACHE_SAMPLE_SEED = 'pi-session-recall-issue-112-no-cache-v1';

/** One physical source selected by the deterministic size-stratified sampler. */
export interface RecallBenchmarkSampleFile {
  sourcePath: string;
  snapshotPath: string;
  relativePath: string;
  sourceBytesAtSelection: number;
  snapshotBytes: number;
  snapshotSha256: string;
  sizeRank: number;
  stratumIndex: number;
}

/** Content-free population and sample evidence retained by the benchmark report. */
export interface RecallBenchmarkSampleSummary {
  seed: string;
  requestedRate: number;
  populationFiles: number;
  populationBytes: number;
  selectedFiles: number;
  selectedBytes: number;
  selectedFileRate: number;
  selectedByteRate: number;
  populationSizeQuantiles: Record<string, number>;
  sampleSizeQuantiles: Record<string, number>;
  sampleIdentitySha256: string;
}

/** Frozen private sample plus aggregate evidence safe to include in a content-free report. */
export interface FrozenRecallBenchmarkSample {
  files: RecallBenchmarkSampleFile[];
  summary: RecallBenchmarkSampleSummary;
  privateManifestPath: string;
}

interface PhysicalSessionFileMetadata {
  sourcePath: string;
  relativePath: string;
  size: number;
}

function deterministicStratumOffset(seed: string, stratumIndex: number, width: number): number {
  const digest = createHash('sha256').update(`${seed}\0${stratumIndex}`).digest();
  return digest.readUInt32BE(0) % width;
}

function selectOneFilePerSizeStratum(
  sortedFiles: readonly PhysicalSessionFileMetadata[],
  selectedCount: number,
  seed: string,
): Array<PhysicalSessionFileMetadata & { sizeRank: number; stratumIndex: number }> {
  const selected: Array<PhysicalSessionFileMetadata & { sizeRank: number; stratumIndex: number }> =
    [];
  for (let stratumIndex = 0; stratumIndex < selectedCount; stratumIndex += 1) {
    const start = Math.floor((stratumIndex * sortedFiles.length) / selectedCount);
    const end = Math.floor(((stratumIndex + 1) * sortedFiles.length) / selectedCount);
    const width = Math.max(end - start, 1);
    const sizeRank = start + deterministicStratumOffset(seed, stratumIndex, width);
    const file = sortedFiles[sizeRank];
    if (!file) {
      throw new Error(`Recall no-cache benchmark sample stratum ${stratumIndex} is empty`);
    }
    selected.push({ ...file, sizeRank, stratumIndex });
  }
  return selected;
}

function selectSizeDistributedPilot<T>(files: readonly T[], pilotCount: number): T[] {
  if (files.length <= pilotCount) {
    return [...files];
  }
  return Array.from({ length: pilotCount }, (_, index) => {
    const selectedIndex = Math.round((index * (files.length - 1)) / (pilotCount - 1));
    const file = files[selectedIndex];
    if (!file) {
      throw new Error(`Recall no-cache benchmark pilot selection missing index ${selectedIndex}`);
    }
    return file;
  });
}

async function hashFileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) {
    const value: unknown = chunk;
    if (!Buffer.isBuffer(value)) {
      throw new Error('Recall no-cache benchmark source hash read a non-buffer chunk');
    }
    hash.update(value);
  }
  return hash.digest('hex');
}

function readSizeQuantiles(sortedSizes: readonly number[]): Record<string, number> {
  if (sortedSizes.length === 0) {
    return { minimum: 0, p25: 0, median: 0, p75: 0, p90: 0, p95: 0, p99: 0, maximum: 0 };
  }
  const at = (proportion: number): number =>
    sortedSizes[
      Math.min(sortedSizes.length - 1, Math.floor((sortedSizes.length - 1) * proportion))
    ] ?? 0;
  return {
    minimum: at(0),
    p25: at(0.25),
    median: at(0.5),
    p75: at(0.75),
    p90: at(0.9),
    p95: at(0.95),
    p99: at(0.99),
    maximum: at(1),
  };
}

function assertSafeRelativeSessionPath(path: string): void {
  if (!path || path === '..' || path.startsWith(`..${sep}`) || resolve('/', path) === path) {
    throw new Error(`Recall no-cache benchmark session path escaped its source root: ${path}`);
  }
}

async function copyFrozenSessionFile(sourcePath: string, snapshotPath: string): Promise<void> {
  await mkdir(dirname(snapshotPath), { recursive: true, mode: 0o700 });
  try {
    await copyFile(sourcePath, snapshotPath, constants.COPYFILE_FICLONE_FORCE);
  } catch (error) {
    const code =
      error && typeof error === 'object' && 'code' in error ? Reflect.get(error, 'code') : null;
    if (code !== 'ENOTSUP' && code !== 'EXDEV' && code !== 'EINVAL') {
      throw error;
    }
    await copyFile(sourcePath, snapshotPath);
  }
  await chmod(snapshotPath, 0o400);
}

/** Selects and freezes a reproducible sample without changing any original Pi session file. */
export async function createFrozenSizeStratifiedRecallSample(options: {
  sessionsDirectory: string;
  snapshotDirectory: string;
  sampleRate: number;
  pilotFileCount?: number;
  excludedSourcePaths?: ReadonlySet<string>;
}): Promise<FrozenRecallBenchmarkSample> {
  if (!(options.sampleRate > 0 && options.sampleRate <= 1)) {
    throw new Error(`Recall no-cache benchmark sample rate invalid: ${options.sampleRate}`);
  }
  const sessionsDirectory = resolve(options.sessionsDirectory);
  const snapshotDirectory = resolve(options.snapshotDirectory);
  const excluded = new Set(
    [...(options.excludedSourcePaths ?? [])].map((sourcePath) => resolve(sourcePath)),
  );
  const sessionPaths = (await listRecallConversationSessionFiles(sessionsDirectory)).filter(
    (sourcePath) => !excluded.has(resolve(sourcePath)),
  );
  const population = await Promise.all(
    sessionPaths.map(async (sourcePath): Promise<PhysicalSessionFileMetadata> => {
      const metadata = await stat(sourcePath);
      const relativePath = relative(sessionsDirectory, sourcePath);
      assertSafeRelativeSessionPath(relativePath);
      return { sourcePath, relativePath, size: metadata.size };
    }),
  );
  population.sort(
    (left, right) => left.size - right.size || left.relativePath.localeCompare(right.relativePath),
  );
  const fullSampleCount = Math.max(1, Math.ceil(population.length * options.sampleRate));
  const fullSample = selectOneFilePerSizeStratum(
    population,
    fullSampleCount,
    RECALL_NO_CACHE_SAMPLE_SEED,
  );
  const selected = options.pilotFileCount
    ? selectSizeDistributedPilot(fullSample, options.pilotFileCount)
    : fullSample;

  await mkdir(snapshotDirectory, { recursive: true, mode: 0o700 });
  await chmod(snapshotDirectory, 0o700);
  const frozenFiles: RecallBenchmarkSampleFile[] = [];
  for (const file of selected) {
    const snapshotPath = join(snapshotDirectory, file.relativePath);
    await copyFrozenSessionFile(file.sourcePath, snapshotPath);
    const snapshotMetadata = await stat(snapshotPath);
    frozenFiles.push({
      sourcePath: file.sourcePath,
      snapshotPath,
      relativePath: file.relativePath,
      sourceBytesAtSelection: file.size,
      snapshotBytes: snapshotMetadata.size,
      snapshotSha256: await hashFileSha256(snapshotPath),
      sizeRank: file.sizeRank,
      stratumIndex: file.stratumIndex,
    });
  }

  const populationBytes = population.reduce((total, file) => total + file.size, 0);
  const selectedBytes = frozenFiles.reduce((total, file) => total + file.snapshotBytes, 0);
  const sampleIdentitySha256 = createHash('sha256')
    .update(
      frozenFiles
        .map((file) => `${file.relativePath}\0${file.snapshotBytes}\0${file.snapshotSha256}`)
        .join('\n'),
    )
    .digest('hex');
  const summary: RecallBenchmarkSampleSummary = {
    seed: RECALL_NO_CACHE_SAMPLE_SEED,
    requestedRate: options.sampleRate,
    populationFiles: population.length,
    populationBytes,
    selectedFiles: frozenFiles.length,
    selectedBytes,
    selectedFileRate: frozenFiles.length / population.length,
    selectedByteRate: selectedBytes / populationBytes,
    populationSizeQuantiles: readSizeQuantiles(population.map((file) => file.size)),
    sampleSizeQuantiles: readSizeQuantiles(
      frozenFiles.map((file) => file.snapshotBytes).toSorted((left, right) => left - right),
    ),
    sampleIdentitySha256,
  };
  const privateManifestPath = join(snapshotDirectory, 'private-sample-manifest.json');
  await writeFile(
    privateManifestPath,
    `${JSON.stringify({ summary, files: frozenFiles }, null, 2)}\n`,
    { encoding: 'utf8', mode: 0o600 },
  );
  return { files: frozenFiles, summary, privateManifestPath };
}
