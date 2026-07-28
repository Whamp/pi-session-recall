import { createHash } from 'node:crypto';
import { copyFile, mkdir, open, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

import { loadOctenConversationTokenizer } from '../../src/octen-conversation-tokenizer.js';
import type { ConversationTextTokenizer } from '../../src/session-conversation-index.js';

const WORKER_SCRATCH_ROOT = resolve('.prototype-data/incremental-worker');
const SOURCE_SESSION_DIRECTORY = join(homedir(), '.pi', 'agent', 'sessions');
const SOURCE_TOKENIZER_DIRECTORY = join(homedir(), '.pi', 'agent', 'recall', 'tokenizers');
const TOKENIZER_REVISION = '6e188e3b072c3e3678b235ad84e6e97bcbb71e8f';
const DELTA_TRAILING_RECORD_COUNTS = [10, 50, 200];
const BOUNDARY_WINDOW_BYTES = 4096;

interface DistributionSummary {
  count: number;
  p50: number;
  p95: number;
  maximum: number;
}

interface SessionFileCandidate {
  path: string;
  size: number;
  mtimeMs: number;
}

interface ParsedVisibleDelta {
  recordCount: number;
  visibleTextBytes: number;
  tokenCount: number;
}

export interface IncrementalWorkerMeasurementReport {
  tokenizerLoadMilliseconds: number;
  residentSetSizeBeforeBytes: number;
  residentSetSizeAfterBytes: number;
  samples: Array<{
    sizeBand: 'small' | 'medium' | 'large';
    sourceBytes: number;
    fullPreparationMilliseconds: number;
    requestedTrailingRecordCount: number;
    deltaBytes: number;
    boundaryVerificationMilliseconds: DistributionSummary;
    deltaReadParseTokenizeMilliseconds: DistributionSummary;
    parsedRecordCount: number;
    visibleTextBytes: number;
    tokenCount: number;
    preparationSpeedup: number;
  }>;
  cacheResolution: {
    candidateTextCount: number;
    coldMissAndWriteMilliseconds: number;
    warmHitReadMilliseconds: number;
    cacheHits: number;
    cacheMisses: number;
  };
}

function summarizeDistribution(values: number[]): DistributionSummary {
  const sorted = [...values].sort((left, right) => left - right);
  const percentile = (fraction: number): number => {
    const index = Math.min(Math.ceil(fraction * sorted.length) - 1, sorted.length - 1);
    return sorted[Math.max(index, 0)] ?? 0;
  };
  return {
    count: values.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    maximum: sorted.at(-1) ?? 0,
  };
}

async function listSessionFileCandidates(directory: string): Promise<SessionFileCandidate[]> {
  const candidates: SessionFileCandidate[] = [];
  async function visit(current: string): Promise<void> {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        const fileStats = await stat(path);
        candidates.push({ path, size: fileStats.size, mtimeMs: fileStats.mtimeMs });
      }
    }
  }
  await visit(directory);
  return candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
}

function selectSessionFileBySize(
  candidates: SessionFileCandidate[],
  minimumBytes: number,
  maximumBytes: number,
): SessionFileCandidate {
  const candidate = candidates.find(
    (item) => item.size >= minimumBytes && item.size <= maximumBytes,
  );
  if (!candidate) {
    throw new Error(
      `Prototype worker session sample missing for byte range ${minimumBytes}..${maximumBytes}`,
    );
  }
  return candidate;
}

async function copyTokenizerAssets(): Promise<string> {
  const sourceRevisionDirectory = join(SOURCE_TOKENIZER_DIRECTORY, TOKENIZER_REVISION);
  const destinationRevisionDirectory = join(WORKER_SCRATCH_ROOT, 'tokenizers', TOKENIZER_REVISION);
  await mkdir(destinationRevisionDirectory, { recursive: true });
  await copyFile(
    join(sourceRevisionDirectory, 'tokenizer.json'),
    join(destinationRevisionDirectory, 'tokenizer.json'),
  );
  await copyFile(
    join(sourceRevisionDirectory, 'tokenizer_config.json'),
    join(destinationRevisionDirectory, 'tokenizer_config.json'),
  );
  return join(WORKER_SCRATCH_ROOT, 'tokenizers');
}

function readVisibleTexts(record: unknown): string[] {
  if (!record || typeof record !== 'object') {
    return [];
  }
  const type = Reflect.get(record, 'type');
  if (type === 'compaction' || type === 'branch_summary') {
    const summary = Reflect.get(record, 'summary');
    return typeof summary === 'string' && summary.length > 0 ? [summary] : [];
  }
  if (type !== 'message') {
    return [];
  }
  const message = Reflect.get(record, 'message');
  if (!message || typeof message !== 'object') {
    return [];
  }
  const role = Reflect.get(message, 'role');
  if (role !== 'user' && role !== 'assistant') {
    return [];
  }
  const content = Reflect.get(message, 'content');
  if (typeof content === 'string') {
    return content.length > 0 ? [content] : [];
  }
  if (!Array.isArray(content)) {
    return [];
  }
  const texts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== 'object' || Reflect.get(block, 'type') !== 'text') {
      continue;
    }
    const text = Reflect.get(block, 'text');
    if (typeof text === 'string' && text.length > 0) {
      texts.push(text);
    }
  }
  return texts;
}

function parseAndTokenizeVisibleRecords(
  content: string,
  tokenizer: ConversationTextTokenizer,
): ParsedVisibleDelta {
  let recordCount = 0;
  let visibleTextBytes = 0;
  let tokenCount = 0;
  for (const line of content.split('\n')) {
    if (!line.trim()) {
      continue;
    }
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    recordCount += 1;
    for (const text of readVisibleTexts(record)) {
      visibleTextBytes += Buffer.byteLength(text);
      tokenCount += tokenizer.encodeConversationText(text).ids.length;
    }
  }
  return { recordCount, visibleTextBytes, tokenCount };
}

function findTrailingRecordOffset(content: Buffer, trailingRecordCount: number): number {
  const recordStarts = [0];
  let lineFeedIndex = content.indexOf(0x0a);
  while (lineFeedIndex !== -1) {
    if (lineFeedIndex + 1 < content.length) {
      recordStarts.push(lineFeedIndex + 1);
    }
    lineFeedIndex = content.indexOf(0x0a, lineFeedIndex + 1);
  }
  const startIndex = Math.max(recordStarts.length - trailingRecordCount, 0);
  return recordStarts[startIndex] ?? 0;
}

function hashBoundary(content: Buffer, offset: number): string {
  return createHash('sha256')
    .update(content.subarray(Math.max(offset - BOUNDARY_WINDOW_BYTES, 0), offset))
    .digest('hex');
}

async function readVerifiedDelta(
  path: string,
  offset: number,
  expectedBoundaryFingerprint: string,
): Promise<Buffer> {
  const file = await open(path, 'r');
  try {
    const boundaryStart = Math.max(offset - BOUNDARY_WINDOW_BYTES, 0);
    const boundary = Buffer.alloc(offset - boundaryStart);
    await file.read(boundary, 0, boundary.length, boundaryStart);
    const actualBoundaryFingerprint = createHash('sha256').update(boundary).digest('hex');
    if (actualBoundaryFingerprint !== expectedBoundaryFingerprint) {
      throw new Error('Prototype append boundary fingerprint mismatch');
    }
    const fileStats = await file.stat();
    const delta = Buffer.alloc(Math.max(fileStats.size - offset, 0));
    await file.read(delta, 0, delta.length, offset);
    return delta;
  } finally {
    await file.close();
  }
}

async function measureContentAddressedCache(
  textDigests: string[],
): Promise<IncrementalWorkerMeasurementReport['cacheResolution']> {
  const cacheDirectory = join(WORKER_SCRATCH_ROOT, 'embedding-cache');
  await mkdir(cacheDirectory, { recursive: true });
  const uniqueDigests = [...new Set(textDigests)].slice(0, 128);
  const vectorBytes = Buffer.alloc(2560 * Float32Array.BYTES_PER_ELEMENT, 1);
  const coldStartedAt = performance.now();
  for (const digest of uniqueDigests) {
    await writeFile(join(cacheDirectory, `${digest}.fp32`), vectorBytes);
  }
  const coldMissAndWriteMilliseconds = performance.now() - coldStartedAt;
  const warmStartedAt = performance.now();
  let cacheHits = 0;
  for (const digest of uniqueDigests) {
    const value = await readFile(join(cacheDirectory, `${digest}.fp32`));
    if (value.length === vectorBytes.length) {
      cacheHits += 1;
    }
  }
  return {
    candidateTextCount: uniqueDigests.length,
    coldMissAndWriteMilliseconds,
    warmHitReadMilliseconds: performance.now() - warmStartedAt,
    cacheHits,
    cacheMisses: uniqueDigests.length,
  };
}

/** Measures copied-session append preparation without opening production recall storage. */
export async function measureIncrementalWorkerPrototype(): Promise<IncrementalWorkerMeasurementReport> {
  await rm(WORKER_SCRATCH_ROOT, { recursive: true, force: true });
  await mkdir(join(WORKER_SCRATCH_ROOT, 'sessions'), { recursive: true });
  try {
    const candidates = await listSessionFileCandidates(SOURCE_SESSION_DIRECTORY);
    const selected = [
      {
        sizeBand: 'small' as const,
        source: selectSessionFileBySize(candidates, 100 * 1024, 1024 * 1024),
      },
      {
        sizeBand: 'medium' as const,
        source: selectSessionFileBySize(candidates, 1024 * 1024, 5 * 1024 * 1024),
      },
      {
        sizeBand: 'large' as const,
        source: selectSessionFileBySize(candidates, 10 * 1024 * 1024, 25 * 1024 * 1024),
      },
    ];
    const tokenizerCacheDirectory = await copyTokenizerAssets();
    const residentSetSizeBeforeBytes = process.memoryUsage().rss;
    const tokenizerStartedAt = performance.now();
    const tokenizer = await loadOctenConversationTokenizer({
      cacheDirectory: tokenizerCacheDirectory,
    });
    const tokenizerLoadMilliseconds = performance.now() - tokenizerStartedAt;
    const residentSetSizeAfterBytes = process.memoryUsage().rss;
    const samples: IncrementalWorkerMeasurementReport['samples'] = [];
    const textDigests: string[] = [];

    for (const sample of selected) {
      const copiedPath = join(WORKER_SCRATCH_ROOT, 'sessions', `${sample.sizeBand}.jsonl`);
      await copyFile(sample.source.path, copiedPath);
      const fullContent = await readFile(copiedPath);
      const fullStartedAt = performance.now();
      parseAndTokenizeVisibleRecords(fullContent.toString('utf8'), tokenizer);
      const fullPreparationMilliseconds = performance.now() - fullStartedAt;

      for (const requestedTrailingRecordCount of DELTA_TRAILING_RECORD_COUNTS) {
        const offset = findTrailingRecordOffset(fullContent, requestedTrailingRecordCount);
        const boundaryFingerprint = hashBoundary(fullContent, offset);
        const boundaryDurations: number[] = [];
        const preparationDurations: number[] = [];
        let parsed: ParsedVisibleDelta = { recordCount: 0, visibleTextBytes: 0, tokenCount: 0 };
        for (let iteration = 0; iteration < 5; iteration += 1) {
          const boundaryStartedAt = performance.now();
          const delta = await readVerifiedDelta(copiedPath, offset, boundaryFingerprint);
          boundaryDurations.push(performance.now() - boundaryStartedAt);
          const preparationStartedAt = performance.now();
          parsed = parseAndTokenizeVisibleRecords(delta.toString('utf8'), tokenizer);
          preparationDurations.push(performance.now() - preparationStartedAt);
        }
        const deltaContent = fullContent.subarray(offset).toString('utf8');
        for (const line of deltaContent.split('\n')) {
          try {
            for (const text of readVisibleTexts(JSON.parse(line))) {
              textDigests.push(createHash('sha256').update(text).digest('hex'));
            }
          } catch {
            // The copied session can end with an incomplete active append; the worker waits for LF.
          }
        }
        const deltaP50 = summarizeDistribution(preparationDurations).p50;
        samples.push({
          sizeBand: sample.sizeBand,
          sourceBytes: fullContent.length,
          fullPreparationMilliseconds,
          requestedTrailingRecordCount,
          deltaBytes: fullContent.length - offset,
          boundaryVerificationMilliseconds: summarizeDistribution(boundaryDurations),
          deltaReadParseTokenizeMilliseconds: summarizeDistribution(preparationDurations),
          parsedRecordCount: parsed.recordCount,
          visibleTextBytes: parsed.visibleTextBytes,
          tokenCount: parsed.tokenCount,
          preparationSpeedup:
            deltaP50 > 0 ? fullPreparationMilliseconds / deltaP50 : Number.POSITIVE_INFINITY,
        });
      }
    }

    return {
      tokenizerLoadMilliseconds,
      residentSetSizeBeforeBytes,
      residentSetSizeAfterBytes,
      samples,
      cacheResolution: await measureContentAddressedCache(textDigests),
    };
  } finally {
    await rm(WORKER_SCRATCH_ROOT, { recursive: true, force: true });
  }
}
