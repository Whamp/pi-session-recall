import {
  createReadStream,
  existsSync,
  opendirSync,
  readFileSync,
  rmSync,
  statfsSync,
  statSync,
} from 'node:fs';
import { mkdir, opendir, writeFile } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';
import { createInterface } from 'node:readline';
import { performance } from 'node:perf_hooks';
import { DatabaseSync } from 'node:sqlite';
import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecFieldSchema,
} from '@zvec/zvec';
import { createOctenHttpEmbeddingProvider } from '../../src/octen-http-embedding-provider.js';
import { loadRecallConversationConfig } from '../../src/recall-conversation-config.js';

const PRODUCTION_DATA_PATH = '/home/will/.pi/agent/recall';
const PRODUCTION_COLLECTION_PATH = join(PRODUCTION_DATA_PATH, 'zvec');
const PRODUCTION_STATE_PATH = join(PRODUCTION_DATA_PATH, 'index-state.json');
const PROTOTYPE_ROOT = '/home/will/.pi/agent/recall-debug/prototype-dense-only';
const PROTOTYPE_COLLECTION_PATH = join(PROTOTYPE_ROOT, 'zvec');
const REPORT_PATH = join(PROTOTYPE_ROOT, 'prototype-report.json');
const INVOCATION_DATABASE_PATH = join(PROTOTYPE_ROOT, 'invocations.sqlite');
const INVOCATION_ZVEC_PATH = join(PROTOTYPE_ROOT, 'invocations-zvec');
const MAX_PROTOTYPE_BYTES = 6 * 1024 ** 3;
const FREE_SPACE_FLOOR_BYTES = 240 * 1024 ** 3;
const FETCH_BATCH_SIZE = 256;
const UPSERT_BATCH_SIZE = 128;

const OMITTED_DENSE_FIELD_NAMES = new Set([
  'identifierContent',
  'isDenseSearchable',
  'toolCallId',
  'toolName',
  'toolCallEntryId',
  'toolResultEntryId',
  'toolError',
]);

const DENSE_QUERIES = [
  'Why have recent pi-session-recall optimization attempts failed?',
  'How is automatic recall indexing scheduled?',
  'Which corrupted February session files are ignored?',
  'How large is the recall database?',
  'Why would an agent use pi-session-recall instead of searching raw JSONL?',
];

const EXACT_SOURCE_QUERIES = [
  'FtsRocksdbReducer',
  'psr optimize',
  'CT1000P3PSSD8',
  '2026-02-02T18-31-25',
  '--optimize-daily',
];

const INVOCATION_QUERIES = [
  'brain_query',
  '/home/will/.pi/agent/TAILNET.md',
  'http://192.168.0.67:8090/v1',
  'psr optimize',
  'gh issue view 165',
  '--optimize-daily',
];

const OMITTED_ARGUMENT_PAYLOAD_KEYS = new Set([
  'body',
  'code',
  'content',
  'newText',
  'oldText',
  'prompt',
  'script',
  'text',
]);

interface PrototypeReport {
  question: string;
  builtAt?: string;
  build?: Record<string, unknown>;
  denseBenchmark?: Record<string, unknown>;
  sourceBenchmark?: Record<string, unknown>;
  invocationBuild?: Record<string, unknown>;
  invocationBenchmark?: Record<string, unknown>;
  invocationStoreComparison?: Record<string, unknown>;
}

interface InvocationRecord {
  sessionPath: string;
  entryId: string;
  kind: 'tool_call' | 'bash_command';
  toolName: string;
  content: string;
}

interface ComparableInvocationRecord extends InvocationRecord {
  id: string;
}

interface InvocationSearchObservation {
  query: string;
  milliseconds: number[];
  resultIds: string[];
}

interface IndexState {
  sessions: Record<string, { chunks: Array<{ id: string }> }>;
}

function readReport(): PrototypeReport {
  if (!existsSync(REPORT_PATH)) {
    return {
      question:
        'Can dense-only flat Zvec plus source-backed exact retrieval eliminate routine compaction while preserving acceptable recall latency?',
    };
  }
  return JSON.parse(readFileSync(REPORT_PATH, 'utf8')) as PrototypeReport;
}

async function writeReport(report: PrototypeReport): Promise<void> {
  await mkdir(dirname(REPORT_PATH), { recursive: true });
  await writeFile(REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`);
}

function allocatedBytes(root: string): number {
  if (!existsSync(root)) return 0;
  const rootStats = statSync(root);
  if (rootStats.isFile()) return rootStats.blocks * 512;
  let total = 0;
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) continue;
    const directory = requireDirectoryEntries(current);
    for (const entry of directory) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(path);
      } else if (entry.isFile()) {
        try {
          total += statSync(path).blocks * 512;
        } catch {
          // Zvec may rename files while the prototype observes its scratch directory.
        }
      }
    }
  }
  return total;
}

function requireDirectoryEntries(path: string) {
  return [...readDirectorySync(path)];
}

function* readDirectorySync(path: string) {
  const directory = opendirSync(path);
  try {
    while (true) {
      const entry = directory.readSync();
      if (!entry) return;
      yield entry;
    }
  } finally {
    directory.closeSync();
  }
}

function freeSpaceBytes(path: string): number {
  const stats = statfsSync(path);
  return stats.bavail * stats.bsize;
}

function processIo(): { readBytes: number; writeBytes: number } {
  const values = new Map<string, number>();
  for (const line of readFileSync('/proc/self/io', 'utf8').split('\n')) {
    const separator = line.indexOf(':');
    if (separator < 0) continue;
    values.set(line.slice(0, separator), Number(line.slice(separator + 1).trim()));
  }
  return {
    readBytes: values.get('read_bytes') ?? 0,
    writeBytes: values.get('write_bytes') ?? 0,
  };
}

function deviceWrittenBytes(): number {
  const fields = readFileSync('/sys/block/nvme0n1/stat', 'utf8').trim().split(/\s+/u);
  return Number(fields[6]) * 512;
}

function assertPrototypeStorageGuard(): void {
  const size = allocatedBytes(PROTOTYPE_ROOT);
  if (size > MAX_PROTOTYPE_BYTES) {
    throw new Error(`Prototype exceeded ${(MAX_PROTOTYPE_BYTES / 1024 ** 3).toFixed(1)} GiB limit`);
  }
  const free = freeSpaceBytes(PROTOTYPE_ROOT);
  if (free < FREE_SPACE_FLOOR_BYTES) {
    throw new Error(
      `Prototype free space fell below ${(FREE_SPACE_FLOOR_BYTES / 1024 ** 3).toFixed(0)} GiB floor`,
    );
  }
}

function percentile(values: number[], percentileValue: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(percentileValue * sorted.length) - 1);
  return sorted[index] ?? 0;
}

function candidateFieldSchemas(source: ZVecCollection): ZVecFieldSchema[] {
  return source.schema
    .fields()
    .filter((field) => !OMITTED_DENSE_FIELD_NAMES.has(field.name))
    .map((field) => ({
      name: field.name,
      dataType: field.dataType,
      nullable: field.nullable,
    })) as ZVecFieldSchema[];
}

function allStateDocumentIds(state: IndexState): string[] {
  return Object.values(state.sessions).flatMap((session) =>
    session.chunks.map((chunk) => chunk.id),
  );
}

async function buildDenseOnlyPrototype(reset: boolean): Promise<void> {
  if (existsSync(PROTOTYPE_ROOT)) {
    if (!reset) {
      throw new Error(`Prototype already exists at ${PROTOTYPE_ROOT}; pass --reset to replace it`);
    }
    if (!PROTOTYPE_ROOT.endsWith('/prototype-dense-only')) {
      throw new Error(`Refusing to remove unexpected prototype path ${PROTOTYPE_ROOT}`);
    }
    rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  }
  await mkdir(PROTOTYPE_ROOT, { recursive: true });

  const startedAt = new Date().toISOString();
  const start = performance.now();
  const cpuStart = process.cpuUsage();
  const ioStart = processIo();
  const deviceStart = deviceWrittenBytes();
  const state = JSON.parse(readFileSync(PRODUCTION_STATE_PATH, 'utf8')) as IndexState;
  const allIds = allStateDocumentIds(state);
  const source = ZVecOpen(PRODUCTION_COLLECTION_PATH, { readOnly: true });
  const fields = candidateFieldSchemas(source);
  const fieldNames = fields.map((field) => field.name);
  const denseIds: string[] = [];

  console.error(
    `Scanning ${allIds.length.toLocaleString()} production rows for dense documents...`,
  );
  for (let startIndex = 0; startIndex < allIds.length; startIndex += FETCH_BATCH_SIZE) {
    const batch = allIds.slice(startIndex, startIndex + FETCH_BATCH_SIZE);
    const documents = source.fetchSync({
      ids: batch,
      outputFields: ['isDenseSearchable'],
      includeVector: false,
    });
    for (const document of Object.values(documents)) {
      if (document.fields.isDenseSearchable === true) denseIds.push(document.id);
    }
    if (startIndex > 0 && startIndex % 100_000 === 0) {
      console.error(`  scanned ${startIndex.toLocaleString()} rows`);
    }
  }

  const sourceDimensions = source.schema.vector('embedding').dimension;
  if (sourceDimensions === undefined) {
    source.closeSync();
    throw new Error('Production embedding dimension is missing');
  }
  const candidate = ZVecCreateAndOpen(
    PROTOTYPE_COLLECTION_PATH,
    new ZVecCollectionSchema({
      name: 'pi_session_recall_dense_only_prototype',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: sourceDimensions,
        indexParams: {
          indexType: ZVecIndexType.FLAT,
          metricType: ZVecMetricType.IP,
        },
      },
      fields,
    }),
  );

  console.error(
    `Copying ${denseIds.length.toLocaleString()} dense documents without re-embedding...`,
  );
  let copied = 0;
  for (let startIndex = 0; startIndex < denseIds.length; startIndex += UPSERT_BATCH_SIZE) {
    const ids = denseIds.slice(startIndex, startIndex + UPSERT_BATCH_SIZE);
    const documents = source.fetchSync({ ids, outputFields: fieldNames, includeVector: true });
    candidate.upsertSync(
      Object.values(documents).map((document) => {
        const embedding = document.vectors.embedding;
        if (embedding === undefined) {
          throw new Error(`Production dense document ${document.id} has no embedding`);
        }
        return {
          id: document.id,
          vectors: { embedding },
          fields: document.fields,
        };
      }),
    );
    copied += Object.keys(documents).length;
    if (startIndex > 0 && startIndex % 10_000 === 0) {
      assertPrototypeStorageGuard();
      console.error(`  copied ${copied.toLocaleString()} documents`);
    }
  }
  const candidateStats = candidate.stats;
  candidate.closeSync();
  source.closeSync();
  assertPrototypeStorageGuard();

  const cpu = process.cpuUsage(cpuStart);
  const ioEnd = processIo();
  const report = readReport();
  report.builtAt = startedAt;
  report.build = {
    elapsedSeconds: (performance.now() - start) / 1_000,
    userCpuSeconds: cpu.user / 1_000_000,
    systemCpuSeconds: cpu.system / 1_000_000,
    processReadBytes: ioEnd.readBytes - ioStart.readBytes,
    processWriteBytes: ioEnd.writeBytes - ioStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStart,
    sourceDocuments: allIds.length,
    denseDocuments: denseIds.length,
    omittedDocuments: allIds.length - denseIds.length,
    prototypeAllocatedBytes: allocatedBytes(PROTOTYPE_ROOT),
    candidateStats,
    candidateFields: fieldNames,
  };
  await writeReport(report);
  console.log(JSON.stringify(report.build, null, 2));
}

async function benchmarkDenseSearch(): Promise<void> {
  if (!existsSync(PROTOTYPE_COLLECTION_PATH)) {
    throw new Error(`Build the prototype first: ${PROTOTYPE_COLLECTION_PATH} is missing`);
  }
  const config = await loadRecallConversationConfig();
  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const production = ZVecOpen(PRODUCTION_COLLECTION_PATH, { readOnly: true });
  const candidate = ZVecOpen(PROTOTYPE_COLLECTION_PATH, { readOnly: true });
  const observations = [];

  for (const query of DENSE_QUERIES) {
    console.error(`Embedding and benchmarking: ${query}`);
    const embedding = await embeddingProvider.embedQuery(query);
    const productionTimes = [];
    const candidateTimes = [];
    let productionIds: string[] = [];
    let candidateIds: string[] = [];
    for (let repetition = 0; repetition < 6; repetition += 1) {
      let started = performance.now();
      const productionResults = production.querySync({
        fieldName: 'embedding',
        vector: embedding,
        topk: 8,
        outputFields: [],
        includeVector: false,
        filter: 'isDenseSearchable = true',
        params: { indexType: ZVecIndexType.HNSW, ef: 300 },
      });
      const productionElapsed = performance.now() - started;
      started = performance.now();
      const candidateResults = candidate.querySync({
        fieldName: 'embedding',
        vector: embedding,
        topk: 8,
        outputFields: [],
        includeVector: false,
      });
      const candidateElapsed = performance.now() - started;
      if (repetition > 0) {
        productionTimes.push(productionElapsed);
        candidateTimes.push(candidateElapsed);
      }
      productionIds = productionResults.map((result) => result.id);
      candidateIds = candidateResults.map((result) => result.id);
    }
    observations.push({
      query,
      productionHnswMilliseconds: productionTimes,
      candidateFlatMilliseconds: candidateTimes,
      productionTopIds: productionIds,
      candidateTopIds: candidateIds,
      topResultMatches: productionIds[0] === candidateIds[0],
      topEightOverlap: candidateIds.filter((id) => productionIds.includes(id)).length,
    });
  }

  production.closeSync();
  candidate.closeSync();
  const productionTimes = observations.flatMap(
    (observation) => observation.productionHnswMilliseconds,
  );
  const candidateTimes = observations.flatMap(
    (observation) => observation.candidateFlatMilliseconds,
  );
  const report = readReport();
  report.denseBenchmark = {
    observations,
    productionHnswMedianMilliseconds: percentile(productionTimes, 0.5),
    productionHnswP95Milliseconds: percentile(productionTimes, 0.95),
    candidateFlatMedianMilliseconds: percentile(candidateTimes, 0.5),
    candidateFlatP95Milliseconds: percentile(candidateTimes, 0.95),
    matchingTopResults: observations.filter((observation) => observation.topResultMatches).length,
    queryCount: observations.length,
  };
  await writeReport(report);
  console.log(JSON.stringify(report.denseBenchmark, null, 2));
}

function flattenLocatorArguments(
  value: unknown,
  path: string,
  parts: string[],
  remainingCharacters: { value: number },
): void {
  if (remainingCharacters.value <= 0 || value === null || value === undefined) return;
  const key = path.split('.').at(-1) ?? '';
  if (OMITTED_ARGUMENT_PAYLOAD_KEYS.has(key)) return;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    const rendered = `${path}=${String(value)}`.slice(
      0,
      Math.min(1_024, remainingCharacters.value),
    );
    parts.push(rendered);
    remainingCharacters.value -= rendered.length;
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < Math.min(value.length, 20); index += 1) {
      flattenLocatorArguments(value[index], `${path}[${index}]`, parts, remainingCharacters);
    }
    return;
  }
  if (typeof value === 'object') {
    for (const [name, nested] of Object.entries(value as Record<string, unknown>)) {
      flattenLocatorArguments(nested, path ? `${path}.${name}` : name, parts, remainingCharacters);
    }
  }
}

function compactInvocationArguments(argumentsValue: unknown): string {
  const parts: string[] = [];
  flattenLocatorArguments(argumentsValue, '', parts, { value: 4_096 });
  return parts.join(' ');
}

function extractInvocationRecords(
  entry: Record<string, unknown>,
  sessionPath: string,
): InvocationRecord[] {
  const message = entry.message as Record<string, unknown> | undefined;
  if (!message) return [];
  const entryId = typeof entry.id === 'string' ? entry.id : '';
  if (message.role === 'bashExecution' && typeof message.command === 'string') {
    return [
      {
        sessionPath,
        entryId,
        kind: 'bash_command',
        toolName: 'bashExecution',
        content: message.command.slice(0, 4_096),
      },
    ];
  }
  if (message.role !== 'assistant' || !Array.isArray(message.content)) return [];
  const records: InvocationRecord[] = [];
  for (const block of message.content) {
    if (!block || typeof block !== 'object') continue;
    const candidate = block as Record<string, unknown>;
    if (candidate.type !== 'toolCall' || typeof candidate.name !== 'string') continue;
    const argumentsText = compactInvocationArguments(candidate.arguments);
    records.push({
      sessionPath,
      entryId,
      kind: 'tool_call',
      toolName: candidate.name,
      content: argumentsText ? `${candidate.name} ${argumentsText}` : candidate.name,
    });
  }
  return records;
}

async function listJsonlFiles(root: string): Promise<string[]> {
  const paths: string[] = [];
  const directory = await opendir(root, { recursive: true });
  for await (const entry of directory) {
    if (entry.isFile() && entry.name.endsWith('.jsonl')) {
      paths.push(join(entry.parentPath, entry.name));
    }
  }
  return paths.sort();
}

function classifyMatchingEntry(entry: Record<string, unknown>): string {
  const message = entry.message as Record<string, unknown> | undefined;
  const role = message?.role;
  if (role === 'toolResult') return 'tool_result';
  if (role === 'bashExecution') return 'bash_command_or_output';
  if (role === 'assistant') return 'assistant_or_tool_call';
  return 'other_source';
}

async function readSessionInvocationRecords(path: string): Promise<InvocationRecord[]> {
  const records: InvocationRecord[] = [];
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.includes('"role":"assistant"') && !line.includes('"role":"bashExecution"')) {
      continue;
    }
    try {
      records.push(...extractInvocationRecords(JSON.parse(line) as Record<string, unknown>, path));
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      // The production index already owns malformed-session policy; this prototype skips bad JSON.
    }
  }
  return records;
}

async function collectComparableInvocationRecords(): Promise<{
  records: ComparableInvocationRecord[];
  filesScanned: number;
  elapsedSeconds: number;
  currentSessionPath: string | null;
}> {
  const config = await loadRecallConversationConfig();
  const files = await listJsonlFiles(config.sessionsDirectory);
  const currentSessionPath =
    Object.keys(
      (JSON.parse(readFileSync(PRODUCTION_STATE_PATH, 'utf8')) as IndexState).sessions,
    ).find((path) => path.includes('019fe31e-3e91-7df4-969a-8c8b1f1ec757')) ?? null;
  const started = performance.now();
  const records: ComparableInvocationRecord[] = [];
  for (const path of files) {
    const sessionRecords = await readSessionInvocationRecords(path);
    for (const record of sessionRecords) {
      records.push({
        ...record,
        id: `invocation-${String(records.length + 1).padStart(9, '0')}`,
      });
    }
  }
  return {
    records,
    filesScanned: files.length,
    elapsedSeconds: (performance.now() - started) / 1_000,
    currentSessionPath,
  };
}

function invocationSqliteAllocatedBytes(): number {
  return ['', '-wal', '-shm'].reduce(
    (total, suffix) => total + allocatedBytes(`${INVOCATION_DATABASE_PATH}${suffix}`),
    0,
  );
}

function openInvocationDatabase(): DatabaseSync {
  const database = new DatabaseSync(INVOCATION_DATABASE_PATH);
  database.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
  return database;
}

async function buildInvocationPrototype(): Promise<void> {
  for (const suffix of ['', '-wal', '-shm']) {
    rmSync(`${INVOCATION_DATABASE_PATH}${suffix}`, { force: true });
  }
  const config = await loadRecallConversationConfig();
  const files = await listJsonlFiles(config.sessionsDirectory);
  const started = performance.now();
  const cpuStart = process.cpuUsage();
  const ioStart = processIo();
  const deviceStart = deviceWrittenBytes();
  const database = openInvocationDatabase();
  database.exec(`
    CREATE VIRTUAL TABLE invocations USING fts5(
      tool_name,
      content,
      session_path UNINDEXED,
      entry_id UNINDEXED,
      kind UNINDEXED,
      tokenize = 'unicode61'
    );
    BEGIN IMMEDIATE;
  `);
  const insert = database.prepare(
    'INSERT INTO invocations(tool_name, content, session_path, entry_id, kind) VALUES (?, ?, ?, ?, ?)',
  );
  let invocationCount = 0;
  let currentSessionRecords: InvocationRecord[] = [];
  const currentSessionPath = Object.keys(
    (JSON.parse(readFileSync(PRODUCTION_STATE_PATH, 'utf8')) as IndexState).sessions,
  ).find((path) => path.includes('019fe31e-3e91-7df4-969a-8c8b1f1ec757'));
  for (const path of files) {
    const records = await readSessionInvocationRecords(path);
    for (const record of records) {
      insert.run(record.toolName, record.content, record.sessionPath, record.entryId, record.kind);
      invocationCount += 1;
    }
    if (path === currentSessionPath) currentSessionRecords = records;
  }
  database.exec('COMMIT; PRAGMA wal_checkpoint(TRUNCATE);');
  const buildCpu = process.cpuUsage(cpuStart);
  const buildIo = processIo();
  const buildDeviceWrites = deviceWrittenBytes() - deviceStart;

  const updateCpuStart = process.cpuUsage();
  const updateIoStart = processIo();
  const updateDeviceStart = deviceWrittenBytes();
  if (currentSessionPath) {
    database.exec('BEGIN IMMEDIATE;');
    database.prepare('DELETE FROM invocations WHERE session_path = ?').run(currentSessionPath);
    for (const record of currentSessionRecords) {
      insert.run(record.toolName, record.content, record.sessionPath, record.entryId, record.kind);
    }
    database.exec('COMMIT; PRAGMA wal_checkpoint(TRUNCATE);');
  }
  const updateCpu = process.cpuUsage(updateCpuStart);
  const updateIo = processIo();
  const updateDeviceWrites = deviceWrittenBytes() - updateDeviceStart;
  database.close();
  assertPrototypeStorageGuard();

  const report = readReport();
  report.invocationBuild = {
    elapsedSeconds: (performance.now() - started) / 1_000,
    userCpuSeconds: buildCpu.user / 1_000_000,
    systemCpuSeconds: buildCpu.system / 1_000_000,
    processReadBytes: buildIo.readBytes - ioStart.readBytes,
    processWriteBytes: buildIo.writeBytes - ioStart.writeBytes,
    deviceWrittenBytes: buildDeviceWrites,
    filesScanned: files.length,
    invocationCount,
    databaseAllocatedBytes: allocatedBytes(PROTOTYPE_ROOT),
    changedSessionSimulation: {
      sessionPath: currentSessionPath ?? null,
      invocationCount: currentSessionRecords.length,
      userCpuSeconds: updateCpu.user / 1_000_000,
      systemCpuSeconds: updateCpu.system / 1_000_000,
      processReadBytes: updateIo.readBytes - updateIoStart.readBytes,
      processWriteBytes: updateIo.writeBytes - updateIoStart.writeBytes,
      deviceWrittenBytes: updateDeviceWrites,
    },
  };
  await writeReport(report);
  console.log(JSON.stringify(report.invocationBuild, null, 2));
}

function quoteFtsPhrase(query: string): string {
  return `"${query.replaceAll('"', '""')}"`;
}

async function benchmarkInvocationSearch(): Promise<void> {
  if (!existsSync(INVOCATION_DATABASE_PATH)) {
    throw new Error(`Build the invocation prototype first: ${INVOCATION_DATABASE_PATH} is missing`);
  }
  const database = openInvocationDatabase();
  const statement = database.prepare(`
    SELECT tool_name, content, session_path, entry_id, kind
    FROM invocations
    WHERE invocations MATCH ?
    LIMIT 20
  `);
  const observations = [];
  for (const query of INVOCATION_QUERIES) {
    const times = [];
    let results: unknown[] = [];
    for (let repetition = 0; repetition < 6; repetition += 1) {
      const started = performance.now();
      results = statement.all(quoteFtsPhrase(query));
      const elapsed = performance.now() - started;
      if (repetition > 0) times.push(elapsed);
    }
    observations.push({ query, milliseconds: times, resultCount: results.length, results });
  }
  database.close();
  const times = observations.flatMap((observation) => observation.milliseconds);
  const report = readReport();
  report.invocationBenchmark = {
    observations,
    medianMilliseconds: percentile(times, 0.5),
    p95Milliseconds: percentile(times, 0.95),
  };
  await writeReport(report);
  console.log(JSON.stringify(report.invocationBenchmark, null, 2));
}

function createInvocationZvecSchema(): ZVecCollectionSchema {
  return new ZVecCollectionSchema({
    name: 'pi_session_recall_invocation_only_prototype',
    fields: [
      {
        name: 'content',
        dataType: ZVecDataType.STRING,
        indexParams: {
          indexType: ZVecIndexType.FTS,
          tokenizerName: 'standard',
          filters: ['lowercase'],
        },
      },
      { name: 'toolName', dataType: ZVecDataType.STRING },
      { name: 'sessionPath', dataType: ZVecDataType.STRING },
      { name: 'entryId', dataType: ZVecDataType.STRING },
      { name: 'kind', dataType: ZVecDataType.STRING },
    ],
  });
}

function invocationRecordToZvecDocument(record: ComparableInvocationRecord) {
  return {
    id: record.id,
    vectors: {},
    fields: {
      content: record.content,
      toolName: record.toolName,
      sessionPath: record.sessionPath,
      entryId: record.entryId,
      kind: record.kind,
    },
  };
}

function benchmarkSqliteInvocationQueries(): InvocationSearchObservation[] {
  const database = openInvocationDatabase();
  const statement = database.prepare(`
    SELECT record_id
    FROM invocations
    WHERE invocations MATCH ?
    ORDER BY bm25(invocations), rowid
    LIMIT 20
  `);
  const observations = INVOCATION_QUERIES.map((query) => {
    const milliseconds: number[] = [];
    let resultIds: string[] = [];
    for (let repetition = 0; repetition < 6; repetition += 1) {
      const started = performance.now();
      resultIds = statement
        .all(quoteFtsPhrase(query))
        .map((row) => String((row as { record_id: unknown }).record_id));
      if (repetition > 0) milliseconds.push(performance.now() - started);
    }
    return { query, milliseconds, resultIds };
  });
  database.close();
  return observations;
}

function benchmarkZvecInvocationQueries(): InvocationSearchObservation[] {
  const collection = ZVecOpen(INVOCATION_ZVEC_PATH, { readOnly: true });
  const observations = INVOCATION_QUERIES.map((query) => {
    const milliseconds: number[] = [];
    let resultIds: string[] = [];
    for (let repetition = 0; repetition < 6; repetition += 1) {
      const started = performance.now();
      resultIds = collection
        .querySync({
          fieldName: 'content',
          fts: { queryString: quoteFtsPhrase(query) },
          topk: 20,
          outputFields: [],
          includeVector: false,
          params: { indexType: ZVecIndexType.FTS, defaultOperator: 'AND' },
        })
        .map((result) => result.id);
      if (repetition > 0) milliseconds.push(performance.now() - started);
    }
    return { query, milliseconds, resultIds };
  });
  collection.closeSync();
  return observations;
}

function summarizeInvocationQueryBenchmark(observations: InvocationSearchObservation[]) {
  const milliseconds = observations.flatMap((observation) => observation.milliseconds);
  return {
    medianMilliseconds: percentile(milliseconds, 0.5),
    p95Milliseconds: percentile(milliseconds, 0.95),
    observations: observations.map((observation) => ({
      query: observation.query,
      resultCount: observation.resultIds.length,
      milliseconds: observation.milliseconds,
    })),
  };
}

function compareInvocationSearchResults(
  sqlite: InvocationSearchObservation[],
  zvec: InvocationSearchObservation[],
) {
  return sqlite.map((sqliteObservation) => {
    const zvecObservation = zvec.find((candidate) => candidate.query === sqliteObservation.query);
    if (!zvecObservation) {
      throw new Error(`Zvec Invocation benchmark missing query ${sqliteObservation.query}`);
    }
    return {
      query: sqliteObservation.query,
      sqliteResultCount: sqliteObservation.resultIds.length,
      zvecResultCount: zvecObservation.resultIds.length,
      matchingTopResult: sqliteObservation.resultIds[0] === zvecObservation.resultIds[0],
      topTwentyOverlap: sqliteObservation.resultIds.filter((id) =>
        zvecObservation.resultIds.includes(id),
      ).length,
    };
  });
}

async function compareInvocationStores(reset: boolean): Promise<void> {
  if (existsSync(PROTOTYPE_ROOT) && reset) {
    if (!PROTOTYPE_ROOT.endsWith('/prototype-dense-only')) {
      throw new Error(`Refusing to remove unexpected prototype path ${PROTOTYPE_ROOT}`);
    }
    rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  }
  await mkdir(PROTOTYPE_ROOT, { recursive: true });
  for (const suffix of ['', '-wal', '-shm']) {
    rmSync(`${INVOCATION_DATABASE_PATH}${suffix}`, { force: true });
  }
  rmSync(INVOCATION_ZVEC_PATH, { recursive: true, force: true });
  assertPrototypeStorageGuard();

  console.error('Extracting one shared Invocation corpus for both stores...');
  const corpus = await collectComparableInvocationRecords();
  const currentSessionRecords = corpus.currentSessionPath
    ? corpus.records.filter((record) => record.sessionPath === corpus.currentSessionPath)
    : [];

  const sqliteBuildStart = performance.now();
  const sqliteBuildCpuStart = process.cpuUsage();
  const sqliteBuildIoStart = processIo();
  const sqliteBuildDeviceStart = deviceWrittenBytes();
  let database = openInvocationDatabase();
  database.exec(`
    CREATE VIRTUAL TABLE invocations USING fts5(
      record_id UNINDEXED,
      tool_name,
      content,
      session_path UNINDEXED,
      entry_id UNINDEXED,
      kind UNINDEXED,
      tokenize = 'unicode61'
    );
    BEGIN IMMEDIATE;
  `);
  let insert = database.prepare(
    'INSERT INTO invocations(record_id, tool_name, content, session_path, entry_id, kind) VALUES (?, ?, ?, ?, ?, ?)',
  );
  for (const record of corpus.records) {
    insert.run(
      record.id,
      record.toolName,
      record.content,
      record.sessionPath,
      record.entryId,
      record.kind,
    );
  }
  database.exec('COMMIT; PRAGMA wal_checkpoint(TRUNCATE);');
  database.close();
  const sqliteBuildCpu = process.cpuUsage(sqliteBuildCpuStart);
  const sqliteBuildIo = processIo();
  const sqliteBuild = {
    elapsedSeconds: (performance.now() - sqliteBuildStart) / 1_000,
    userCpuSeconds: sqliteBuildCpu.user / 1_000_000,
    systemCpuSeconds: sqliteBuildCpu.system / 1_000_000,
    processWriteBytes: sqliteBuildIo.writeBytes - sqliteBuildIoStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - sqliteBuildDeviceStart,
    allocatedBytes: invocationSqliteAllocatedBytes(),
  };

  const sqliteUpdateStart = performance.now();
  const sqliteUpdateCpuStart = process.cpuUsage();
  const sqliteUpdateIoStart = processIo();
  const sqliteUpdateDeviceStart = deviceWrittenBytes();
  database = openInvocationDatabase();
  database.exec('BEGIN IMMEDIATE;');
  if (corpus.currentSessionPath) {
    database
      .prepare('DELETE FROM invocations WHERE session_path = ?')
      .run(corpus.currentSessionPath);
  }
  insert = database.prepare(
    'INSERT INTO invocations(record_id, tool_name, content, session_path, entry_id, kind) VALUES (?, ?, ?, ?, ?, ?)',
  );
  for (const record of currentSessionRecords) {
    insert.run(
      record.id,
      record.toolName,
      record.content,
      record.sessionPath,
      record.entryId,
      record.kind,
    );
  }
  database.exec('COMMIT; PRAGMA wal_checkpoint(TRUNCATE);');
  database.close();
  const sqliteUpdateCpu = process.cpuUsage(sqliteUpdateCpuStart);
  const sqliteUpdateIo = processIo();
  const sqliteUpdate = {
    elapsedSeconds: (performance.now() - sqliteUpdateStart) / 1_000,
    userCpuSeconds: sqliteUpdateCpu.user / 1_000_000,
    systemCpuSeconds: sqliteUpdateCpu.system / 1_000_000,
    processWriteBytes: sqliteUpdateIo.writeBytes - sqliteUpdateIoStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - sqliteUpdateDeviceStart,
    allocatedBytes: invocationSqliteAllocatedBytes(),
  };
  const sqliteSearch = benchmarkSqliteInvocationQueries();

  const zvecBuildStart = performance.now();
  const zvecBuildCpuStart = process.cpuUsage();
  const zvecBuildIoStart = processIo();
  const zvecBuildDeviceStart = deviceWrittenBytes();
  let zvec = ZVecCreateAndOpen(INVOCATION_ZVEC_PATH, createInvocationZvecSchema());
  for (let startIndex = 0; startIndex < corpus.records.length; startIndex += UPSERT_BATCH_SIZE) {
    zvec.upsertSync(
      corpus.records
        .slice(startIndex, startIndex + UPSERT_BATCH_SIZE)
        .map(invocationRecordToZvecDocument),
    );
    if (startIndex > 0 && startIndex % 20_000 === 0) {
      assertPrototypeStorageGuard();
      console.error(`  Zvec inserted ${startIndex.toLocaleString()} Invocation records`);
    }
  }
  const zvecDocCount = zvec.stats.docCount;
  zvec.closeSync();
  const zvecBuildCpu = process.cpuUsage(zvecBuildCpuStart);
  const zvecBuildIo = processIo();
  const zvecBuild = {
    elapsedSeconds: (performance.now() - zvecBuildStart) / 1_000,
    userCpuSeconds: zvecBuildCpu.user / 1_000_000,
    systemCpuSeconds: zvecBuildCpu.system / 1_000_000,
    processWriteBytes: zvecBuildIo.writeBytes - zvecBuildIoStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - zvecBuildDeviceStart,
    allocatedBytes: allocatedBytes(INVOCATION_ZVEC_PATH),
    docCount: zvecDocCount,
  };

  const zvecUpdateStart = performance.now();
  const zvecUpdateCpuStart = process.cpuUsage();
  const zvecUpdateIoStart = processIo();
  const zvecUpdateDeviceStart = deviceWrittenBytes();
  zvec = ZVecOpen(INVOCATION_ZVEC_PATH);
  if (currentSessionRecords.length > 0) {
    zvec.deleteSync(currentSessionRecords.map((record) => record.id));
    for (
      let startIndex = 0;
      startIndex < currentSessionRecords.length;
      startIndex += UPSERT_BATCH_SIZE
    ) {
      zvec.upsertSync(
        currentSessionRecords
          .slice(startIndex, startIndex + UPSERT_BATCH_SIZE)
          .map(invocationRecordToZvecDocument),
      );
    }
  }
  zvec.closeSync();
  const zvecUpdateCpu = process.cpuUsage(zvecUpdateCpuStart);
  const zvecUpdateIo = processIo();
  const zvecUpdate = {
    elapsedSeconds: (performance.now() - zvecUpdateStart) / 1_000,
    userCpuSeconds: zvecUpdateCpu.user / 1_000_000,
    systemCpuSeconds: zvecUpdateCpu.system / 1_000_000,
    processWriteBytes: zvecUpdateIo.writeBytes - zvecUpdateIoStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - zvecUpdateDeviceStart,
    allocatedBytes: allocatedBytes(INVOCATION_ZVEC_PATH),
  };
  const zvecSearchBeforeOptimize = benchmarkZvecInvocationQueries();

  const zvecOptimizeStart = performance.now();
  const zvecOptimizeCpuStart = process.cpuUsage();
  const zvecOptimizeIoStart = processIo();
  const zvecOptimizeDeviceStart = deviceWrittenBytes();
  let zvecPeakAllocatedBytes = allocatedBytes(INVOCATION_ZVEC_PATH);
  zvec = ZVecOpen(INVOCATION_ZVEC_PATH);
  const allocationMonitor = setInterval(() => {
    zvecPeakAllocatedBytes = Math.max(zvecPeakAllocatedBytes, allocatedBytes(INVOCATION_ZVEC_PATH));
  }, 250);
  try {
    await zvec.optimize();
  } finally {
    clearInterval(allocationMonitor);
    zvec.closeSync();
  }
  const zvecOptimizeCpu = process.cpuUsage(zvecOptimizeCpuStart);
  const zvecOptimizeIo = processIo();
  const zvecOptimize = {
    elapsedSeconds: (performance.now() - zvecOptimizeStart) / 1_000,
    userCpuSeconds: zvecOptimizeCpu.user / 1_000_000,
    systemCpuSeconds: zvecOptimizeCpu.system / 1_000_000,
    processWriteBytes: zvecOptimizeIo.writeBytes - zvecOptimizeIoStart.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - zvecOptimizeDeviceStart,
    allocatedBytes: allocatedBytes(INVOCATION_ZVEC_PATH),
    peakAllocatedBytes: zvecPeakAllocatedBytes,
  };
  const zvecSearchAfterOptimize = benchmarkZvecInvocationQueries();
  assertPrototypeStorageGuard();

  const report = readReport();
  report.invocationStoreComparison = {
    measuredAt: new Date().toISOString(),
    corpus: {
      filesScanned: corpus.filesScanned,
      invocationCount: corpus.records.length,
      extractionElapsedSeconds: corpus.elapsedSeconds,
      changedSessionPath: corpus.currentSessionPath,
      changedSessionInvocationCount: currentSessionRecords.length,
    },
    sqlite: {
      build: sqliteBuild,
      changedSessionReplacement: sqliteUpdate,
      search: summarizeInvocationQueryBenchmark(sqliteSearch),
    },
    zvec: {
      vectorFields: 0,
      build: zvecBuild,
      changedSessionReplacement: zvecUpdate,
      searchBeforeOptimize: summarizeInvocationQueryBenchmark(zvecSearchBeforeOptimize),
      optimize: zvecOptimize,
      searchAfterOptimize: summarizeInvocationQueryBenchmark(zvecSearchAfterOptimize),
    },
    resultComparisonBeforeOptimize: compareInvocationSearchResults(
      sqliteSearch,
      zvecSearchBeforeOptimize,
    ),
    zvecResultComparisonAfterOptimize: compareInvocationSearchResults(
      zvecSearchBeforeOptimize,
      zvecSearchAfterOptimize,
    ),
  };
  await writeReport(report);
  console.log(JSON.stringify(report.invocationStoreComparison, null, 2));
}

async function benchmarkExactSourceSearch(): Promise<void> {
  const config = await loadRecallConversationConfig();
  const files = await listJsonlFiles(config.sessionsDirectory);
  const queries = EXACT_SOURCE_QUERIES.map((query) => ({ query, lower: query.toLowerCase() }));
  const matches = new Map(
    queries.map(({ query }) => [query, [] as Array<Record<string, unknown>>]),
  );
  const started = performance.now();
  const cpuStart = process.cpuUsage();
  const ioStart = processIo();
  let bytesScanned = 0;
  let linesScanned = 0;

  for (const path of files) {
    bytesScanned += statSync(path).size;
    const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
    let lineNumber = 0;
    for await (const line of lines) {
      lineNumber += 1;
      linesScanned += 1;
      const lowerLine = line.toLowerCase();
      const matchedQueries = queries.filter(({ lower }) => lowerLine.includes(lower));
      if (matchedQueries.length === 0) continue;
      let entry: Record<string, unknown>;
      try {
        entry = JSON.parse(line) as Record<string, unknown>;
      } catch {
        continue;
      }
      const category = classifyMatchingEntry(entry);
      for (const { query } of matchedQueries) {
        const results = matches.get(query);
        if (!results || results.length >= 20) continue;
        results.push({
          sessionPath: path,
          relativeSessionPath: relative(config.sessionsDirectory, path),
          lineNumber,
          entryId: entry.id ?? null,
          category,
        });
      }
    }
  }

  const cpu = process.cpuUsage(cpuStart);
  const ioEnd = processIo();
  const report = readReport();
  report.sourceBenchmark = {
    elapsedSeconds: (performance.now() - started) / 1_000,
    userCpuSeconds: cpu.user / 1_000_000,
    systemCpuSeconds: cpu.system / 1_000_000,
    processReadBytes: ioEnd.readBytes - ioStart.readBytes,
    processWriteBytes: ioEnd.writeBytes - ioStart.writeBytes,
    filesScanned: files.length,
    linesScanned,
    bytesScanned,
    queries: Object.fromEntries(matches),
  };
  await writeReport(report);
  console.log(JSON.stringify(report.sourceBenchmark, null, 2));
}

async function main(): Promise<void> {
  const command = process.argv[2];
  const reset = process.argv.includes('--reset');
  const usage =
    'Usage: npm run prototype:recall-storage-layout -- <build|benchmark-dense|build-invocations|benchmark-invocations|compare-invocation-stores|benchmark-source|run> [--reset]';
  if (command === '--help') {
    console.log(usage);
    return;
  }
  if (command === 'build') {
    await buildDenseOnlyPrototype(reset);
    return;
  }
  if (command === 'benchmark-dense') {
    await benchmarkDenseSearch();
    return;
  }
  if (command === 'benchmark-source') {
    await benchmarkExactSourceSearch();
    return;
  }
  if (command === 'build-invocations') {
    await buildInvocationPrototype();
    return;
  }
  if (command === 'benchmark-invocations') {
    await benchmarkInvocationSearch();
    return;
  }
  if (command === 'compare-invocation-stores') {
    await compareInvocationStores(reset);
    return;
  }
  if (command === 'run') {
    await buildDenseOnlyPrototype(reset);
    await benchmarkDenseSearch();
    await buildInvocationPrototype();
    await benchmarkInvocationSearch();
    await benchmarkExactSourceSearch();
    return;
  }
  throw new Error(usage);
}

await main();
