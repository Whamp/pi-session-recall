import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statfsSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  DatabaseSync,
  type SQLInputValue,
  type SQLOutputValue,
  type StatementSync,
} from 'node:sqlite';
import { ZVecOpen, type ZVecCollection } from '@zvec/zvec';
import * as sqliteVec from 'sqlite-vec';

import { createOctenHttpEmbeddingProvider } from '../../src/octen-http-embedding-provider.js';
import { loadRecallConversationConfig } from '../../src/recall-conversation-config.js';

const RECALL_DATA_PATH = '/home/will/.pi/agent/recall';
const RECALL_GENERATIONS_PATH = join(RECALL_DATA_PATH, 'generations');
const PROTOTYPE_ROOT = '/home/will/.pi/agent/recall-debug/prototype-sqlite-vec';
const CANDIDATE_DATABASE_PATH = join(PROTOTYPE_ROOT, 'recall-all.sqlite');
const PARTITIONED_VECTOR_TABLE = 'dense_embeddings_partitioned';
const BUCKETED_VECTOR_TABLE = 'dense_embeddings_bucketed';
const PROJECT_BUCKET_COUNT = 16;
const REPORT_PATH = join(PROTOTYPE_ROOT, 'sqlite-vec-report.json');
const CRASH_MARKER_PATH = join(PROTOTYPE_ROOT, 'crash-child-mutated');
const MAX_PROTOTYPE_BYTES = 6 * 1024 ** 3;
const FREE_SPACE_FLOOR_BYTES = 240 * 1024 ** 3;
const FETCH_BATCH_SIZE = 128;
const DENSE_SEARCH_LIMIT = 8;
const SQLITE_CACHE_KIBIBYTES = 64 * 1024;
const SQLITE_MMAP_BYTES = 2 * 1024 ** 3;
const DEFAULT_CHURN_CYCLES = 100;
const REPRESENTATIVE_INVOCATION_COUNT = 814;
const PROJECT_IDENTITY = 'git-origin:github.com/Whamp/pi-session-recall';
const PROJECT_IDENTITY_DIGEST = createHash('sha256').update(PROJECT_IDENTITY).digest('hex');

const DENSE_QUERIES = [
  'Why have recent pi-session-recall optimization attempts failed?',
  'How is automatic recall indexing scheduled?',
  'Which corrupted February session files are ignored?',
  'How large is the recall database?',
  'Why would an agent use pi-session-recall instead of searching raw JSONL?',
];

const INVOCATION_QUERIES = [
  'brain_query',
  '/home/will/.pi/agent/TAILNET.md',
  'http://192.168.0.67:8090/v1',
  'psr optimize',
  'gh issue view 165',
  '--optimize-daily',
];

interface SqliteVecPrototypeReport {
  question: string;
  source?: Record<string, unknown>;
  build?: Record<string, unknown>;
  benchmarkBeforeChurn?: Record<string, unknown>;
  atomicity?: Record<string, unknown>;
  changedSessionUpdate?: Record<string, unknown>;
  churn?: Record<string, unknown>;
  benchmarkAfterChurn?: Record<string, unknown>;
  partitionedAlternativeBuild?: Record<string, unknown>;
  partitionedAlternativeBenchmark?: Record<string, unknown>;
  bucketedAlternativeBuild?: Record<string, unknown>;
  bucketedAlternativeBenchmark?: Record<string, unknown>;
  limitations?: string[];
}

interface SourceGenerationPaths {
  root: string;
  catalog: string;
  denseCollection: string;
  manifest: string;
}

interface DenseFieldSchema {
  name: string;
  dataType: number;
}

interface DenseBenchmarkObservation {
  scope: 'global' | 'project';
  query: string;
  sourceZvecMilliseconds: number[];
  sqliteVecMilliseconds: number[];
  sourceTopIds: string[];
  sqliteTopIds: string[];
  topResultMatches: boolean;
  topEightOverlap: number;
}

interface PartitionedDenseBenchmarkObservation {
  scope: 'global' | 'project';
  query: string;
  sourceZvecMilliseconds: number[];
  metadataSqliteVecMilliseconds: number[];
  partitionedSqliteVecMilliseconds: number[];
  sourceTopIds: string[];
  metadataTopIds: string[];
  partitionedTopIds: string[];
  partitionedTopResultMatches: boolean;
  partitionedTopEightOverlap: number;
}

interface BucketedDenseBenchmarkObservation {
  scope: 'global' | 'project';
  query: string;
  sourceZvecMilliseconds: number[];
  metadataSqliteVecMilliseconds: number[];
  bucketedSqliteVecMilliseconds: number[];
  sourceTopIds: string[];
  metadataTopIds: string[];
  bucketedTopIds: string[];
  bucketedTopResultMatches: boolean;
  bucketedTopEightOverlap: number;
}

type SqliteRow = Record<string, SQLOutputValue>;

interface CandidateSessionData {
  sessionPath: string;
  sessionDocumentColumns: string[];
  sessionDocuments: SqliteRow[];
  invocationColumns: string[];
  invocations: SqliteRow[];
  denseDocumentColumns: string[];
  denseDocumentsToRewrite: SqliteRow[];
  denseVectorsToRewrite: SqliteRow[];
}

function readReport(): SqliteVecPrototypeReport {
  if (!existsSync(REPORT_PATH)) {
    return {
      question:
        'Can one SQLite database with FTS5 and sqlite-vec supersede flat Zvec plus a SQLite catalog for compact session recall?',
    };
  }
  return JSON.parse(readFileSync(REPORT_PATH, 'utf8')) as SqliteVecPrototypeReport;
}

async function writeReport(report: SqliteVecPrototypeReport): Promise<void> {
  await writeFile(REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`);
}

function resolveSourceGeneration(): SourceGenerationPaths {
  const candidates = readdirSync(RECALL_GENERATIONS_PATH, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith('generation-'))
    .map((entry) => {
      const root = join(RECALL_GENERATIONS_PATH, entry.name);
      return {
        root,
        catalog: join(root, 'recall-catalog.sqlite'),
        denseCollection: join(root, 'zvec'),
        manifest: join(root, 'index-manifest.json'),
      };
    })
    .filter(
      (candidate) =>
        existsSync(candidate.catalog) &&
        existsSync(candidate.denseCollection) &&
        existsSync(candidate.manifest),
    );
  if (candidates.length !== 1) {
    throw new Error(
      `SQLite vec prototype source generation ambiguous: expected exactly one staged generation, found ${candidates.length}`,
    );
  }
  const source = candidates[0];
  if (!source) throw new Error('SQLite vec prototype source generation missing');
  return source;
}

function allocatedBytes(path: string): number {
  if (!existsSync(path)) return 0;
  const stats = statSync(path);
  if (stats.isFile()) return stats.blocks * 512;
  let total = 0;
  for (const entry of readdirSync(path, { withFileTypes: true })) {
    total += allocatedBytes(join(path, entry.name));
  }
  return total;
}

function candidateAllocatedBytes(): number {
  return ['', '-wal', '-shm'].reduce(
    (total, suffix) => total + allocatedBytes(`${CANDIDATE_DATABASE_PATH}${suffix}`),
    0,
  );
}

function freeSpaceBytes(path: string): number {
  const stats = statfsSync(path);
  return stats.bavail * stats.bsize;
}

function assertPrototypeStorageGuard(): void {
  const size = allocatedBytes(PROTOTYPE_ROOT);
  if (size > MAX_PROTOTYPE_BYTES) {
    throw new Error(
      `SQLite vec prototype exceeded ${(MAX_PROTOTYPE_BYTES / 1024 ** 3).toFixed(1)} GiB allocation limit`,
    );
  }
  const free = freeSpaceBytes(PROTOTYPE_ROOT);
  if (free < FREE_SPACE_FLOOR_BYTES) {
    throw new Error(
      `SQLite vec prototype free space fell below ${(FREE_SPACE_FLOOR_BYTES / 1024 ** 3).toFixed(0)} GiB floor`,
    );
  }
}

function processIo(): { readBytes: number; writeBytes: number } {
  const values = new Map<string, number>();
  for (const line of readFileSync('/proc/self/io', 'utf8').split('\n')) {
    const separator = line.indexOf(':');
    if (separator >= 0) {
      values.set(line.slice(0, separator), Number(line.slice(separator + 1).trim()));
    }
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

function processPeakResidentBytes(): number {
  const match = readFileSync('/proc/self/status', 'utf8').match(/^VmHWM:\s+(\d+)\s+kB$/mu);
  return match ? Number(match[1]) * 1024 : 0;
}

function percentile(values: number[], percentileValue: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(percentileValue * sorted.length) - 1);
  return sorted[index] ?? 0;
}

function quoteSqlIdentifier(identifier: string): string {
  return `"${identifier.replaceAll('"', '""')}"`;
}

function quoteSqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function quoteFtsPhrase(query: string): string {
  return `"${query.replaceAll('"', '""')}"`;
}

function sqliteTypeForZvecField(dataType: number): 'INTEGER' | 'TEXT' {
  if (dataType === 3 || dataType === 4) return 'INTEGER';
  return 'TEXT';
}

function sqliteValueForZvecField(dataType: number, value: unknown): SQLInputValue {
  if (value === null || value === undefined) return null;
  if (dataType === 41) return JSON.stringify(value);
  if (dataType === 3) return value === true ? 1 : 0;
  if (typeof value === 'number' || typeof value === 'string') return value;
  throw new Error(`SQLite vec prototype cannot encode Zvec field type ${dataType}`);
}

function embeddingBlob(embedding: unknown): Uint8Array {
  if (embedding instanceof Float32Array) {
    return new Uint8Array(embedding.buffer, embedding.byteOffset, embedding.byteLength);
  }
  if (Array.isArray(embedding)) {
    const values = Float32Array.from(embedding as number[], Math.fround);
    return new Uint8Array(values.buffer);
  }
  throw new Error('SQLite vec prototype source embedding missing or invalid');
}

function configureCandidateDatabase(database: DatabaseSync, writable: boolean): void {
  sqliteVec.load(database);
  database.exec(`
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;
    PRAGMA cache_size = -${SQLITE_CACHE_KIBIBYTES};
    PRAGMA mmap_size = ${SQLITE_MMAP_BYTES};
    PRAGMA temp_store = MEMORY;
  `);
  if (writable) {
    database.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
  } else {
    database.exec('PRAGMA query_only = ON;');
  }
}

function openCandidateDatabase(writable: boolean): DatabaseSync {
  const database = new DatabaseSync(CANDIDATE_DATABASE_PATH, {
    allowExtension: true,
    readOnly: !writable,
  });
  configureCandidateDatabase(database, writable);
  return database;
}

function createDenseSqliteSchema(
  database: DatabaseSync,
  fields: readonly DenseFieldSchema[],
): void {
  const fieldColumns = fields
    .map(
      (field) =>
        `${quoteSqlIdentifier(field.name)} ${sqliteTypeForZvecField(field.dataType)} NOT NULL`,
    )
    .join(',\n');
  database.exec(`
    CREATE TABLE dense_projects (
      project_key INTEGER PRIMARY KEY,
      project_identity_digest TEXT NOT NULL UNIQUE
    ) STRICT;

    CREATE TABLE dense_documents (
      rowid INTEGER PRIMARY KEY,
      document_id TEXT NOT NULL UNIQUE,
      project_key INTEGER NOT NULL REFERENCES dense_projects(project_key),
      ${fieldColumns}
    ) STRICT;

    CREATE INDEX dense_documents_session_path_index
      ON dense_documents(${quoteSqlIdentifier('sessionPath')});
    CREATE INDEX dense_documents_project_key_index
      ON dense_documents(project_key);

    CREATE TABLE prototype_dense_field_schema (
      ordinal INTEGER PRIMARY KEY,
      field_name TEXT NOT NULL UNIQUE,
      zvec_data_type INTEGER NOT NULL
    ) STRICT;

    CREATE VIRTUAL TABLE dense_embeddings USING vec0(
      embedding FLOAT[1024] DISTANCE_METRIC=cosine,
      project_key INTEGER
    );
  `);
  const insertField = database.prepare(
    'INSERT INTO prototype_dense_field_schema(ordinal, field_name, zvec_data_type) VALUES (?, ?, ?)',
  );
  for (const [ordinal, field] of fields.entries()) {
    insertField.run(ordinal, field.name, field.dataType);
  }
}

function createDenseDocumentInsert(
  database: DatabaseSync,
  fields: readonly DenseFieldSchema[],
): StatementSync {
  const columns = ['rowid', 'document_id', 'project_key', ...fields.map((field) => field.name)];
  return database.prepare(
    `INSERT INTO dense_documents(${columns.map(quoteSqlIdentifier).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`,
  );
}

function candidateDatabaseMetrics(database: DatabaseSync): Record<string, unknown> {
  const pageCount = Number(database.prepare('PRAGMA page_count').get()?.page_count ?? 0);
  const freePages = Number(database.prepare('PRAGMA freelist_count').get()?.freelist_count ?? 0);
  const pageSize = Number(database.prepare('PRAGMA page_size').get()?.page_size ?? 0);
  return {
    fileBytes: existsSync(CANDIDATE_DATABASE_PATH) ? statSync(CANDIDATE_DATABASE_PATH).size : 0,
    allocatedBytes: candidateAllocatedBytes(),
    pageSize,
    pageCount,
    freePages,
    freeBytes: freePages * pageSize,
    walBytes: existsSync(`${CANDIDATE_DATABASE_PATH}-wal`)
      ? statSync(`${CANDIDATE_DATABASE_PATH}-wal`).size
      : 0,
  };
}

async function buildSqliteVecCandidate(reset: boolean): Promise<void> {
  if (existsSync(PROTOTYPE_ROOT)) {
    if (!reset) {
      throw new Error(
        `SQLite vec prototype exists at ${PROTOTYPE_ROOT}; pass --reset to replace it`,
      );
    }
    if (!PROTOTYPE_ROOT.endsWith('/prototype-sqlite-vec')) {
      throw new Error(`SQLite vec prototype refusing to remove unexpected path ${PROTOTYPE_ROOT}`);
    }
    rmSync(PROTOTYPE_ROOT, { recursive: true, force: true });
  }
  mkdirSync(PROTOTYPE_ROOT, { recursive: true });
  assertPrototypeStorageGuard();

  const sourcePaths = resolveSourceGeneration();
  const sourceManifest = JSON.parse(readFileSync(sourcePaths.manifest, 'utf8')) as Record<
    string,
    unknown
  >;
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const cpuStarted = process.cpuUsage();
  const ioStarted = processIo();
  const deviceStarted = deviceWrittenBytes();

  console.error('Copying the certified v7 SQLite catalog with VACUUM INTO...');
  const sourceCatalog = new DatabaseSync(sourcePaths.catalog, { readOnly: true });
  sourceCatalog.exec(`VACUUM INTO ${quoteSqlString(CANDIDATE_DATABASE_PATH)}`);
  const sourceCatalogCounts = {
    sessions: Number(
      sourceCatalog.prepare('SELECT count(*) AS count FROM physical_sessions').get()?.count ?? 0,
    ),
    invocations: Number(
      sourceCatalog.prepare('SELECT count(*) AS count FROM invocations').get()?.count ?? 0,
    ),
    sessionDocuments: Number(
      sourceCatalog.prepare('SELECT count(*) AS count FROM session_documents').get()?.count ?? 0,
    ),
  };
  sourceCatalog.close();
  assertPrototypeStorageGuard();

  const sourceCollection = ZVecOpen(sourcePaths.denseCollection, { readOnly: true });
  const fields = sourceCollection.schema.fields().map((field) => ({
    name: field.name,
    dataType: field.dataType,
  }));
  const fieldNames = fields.map((field) => field.name);
  const sourceDimensions = sourceCollection.schema.vector('embedding').dimension;
  if (sourceDimensions !== 1_024) {
    sourceCollection.closeSync();
    throw new Error(
      `SQLite vec prototype expected 1024 source dimensions, received ${String(sourceDimensions)}`,
    );
  }

  const database = openCandidateDatabase(true);
  database.exec('PRAGMA wal_autocheckpoint = 0;');
  createDenseSqliteSchema(database, fields);
  const denseDocumentInsert = createDenseDocumentInsert(database, fields);
  const denseVectorInsert = database.prepare(
    'INSERT INTO dense_embeddings(rowid, embedding, project_key) VALUES (?, ?, ?)',
  );
  const projectInsert = database.prepare(
    'INSERT INTO dense_projects(project_key, project_identity_digest) VALUES (?, ?)',
  );
  const sourceState = new DatabaseSync(sourcePaths.catalog, { readOnly: true });
  const documentIds = sourceState.prepare(
    'SELECT document_id FROM session_documents WHERE is_dense = 1 ORDER BY document_id',
  );
  const projectKeys = new Map<string, number>();
  let nextProjectKey = 1;
  let nextRowid = 1;
  let copiedDocuments = 0;
  let transactionOpen = false;

  console.error(
    `Copying ${sourceCollection.stats.docCount.toLocaleString()} dense documents and vectors into sqlite-vec...`,
  );
  try {
    database.exec('BEGIN IMMEDIATE;');
    transactionOpen = true;
    let idBatch: string[] = [];
    const copyBatch = (ids: string[]): void => {
      const documents = sourceCollection.fetchSync({
        ids,
        outputFields: fieldNames,
        includeVector: true,
      });
      for (const document of Object.values(documents)) {
        const digest = String(document.fields.projectIdentityDigest ?? '');
        let projectKey = projectKeys.get(digest);
        if (projectKey === undefined) {
          projectKey = nextProjectKey;
          nextProjectKey += 1;
          projectKeys.set(digest, projectKey);
          projectInsert.run(projectKey, digest);
        }
        const rowid = nextRowid;
        nextRowid += 1;
        denseDocumentInsert.run(
          rowid,
          document.id,
          projectKey,
          ...fields.map((field) =>
            sqliteValueForZvecField(field.dataType, document.fields[field.name]),
          ),
        );
        denseVectorInsert.run(
          BigInt(rowid),
          embeddingBlob(document.vectors.embedding),
          BigInt(projectKey),
        );
        copiedDocuments += 1;
      }
    };

    for (const row of documentIds.iterate()) {
      if (typeof row.document_id !== 'string') {
        throw new Error('SQLite vec prototype source document ID invalid');
      }
      idBatch.push(row.document_id);
      if (idBatch.length === FETCH_BATCH_SIZE) {
        copyBatch(idBatch);
        idBatch = [];
        if (copiedDocuments % 10_000 < FETCH_BATCH_SIZE) {
          assertPrototypeStorageGuard();
          console.error(
            `  copied ${copiedDocuments.toLocaleString()} / ${sourceCollection.stats.docCount.toLocaleString()}`,
          );
        }
      }
    }
    if (idBatch.length > 0) copyBatch(idBatch);
    database.exec('COMMIT;');
    transactionOpen = false;
  } catch (error) {
    if (transactionOpen) database.exec('ROLLBACK;');
    database.close();
    sourceState.close();
    sourceCollection.closeSync();
    throw error;
  }

  const beforeCheckpoint = candidateDatabaseMetrics(database);
  const checkpointStarted = performance.now();
  const checkpointIoStarted = processIo();
  const checkpointDeviceStarted = deviceWrittenBytes();
  database.exec('PRAGMA wal_checkpoint(TRUNCATE); PRAGMA wal_autocheckpoint = 1000;');
  const checkpointIoFinished = processIo();
  const checkpoint = {
    elapsedSeconds: (performance.now() - checkpointStarted) / 1_000,
    processWriteBytes: checkpointIoFinished.writeBytes - checkpointIoStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - checkpointDeviceStarted,
  };
  const integrity = String(database.prepare('PRAGMA integrity_check').get()?.integrity_check);
  const foreignKeyViolations = database.prepare('PRAGMA foreign_key_check').all().length;
  const counts = {
    denseDocuments: Number(
      database.prepare('SELECT count(*) AS count FROM dense_documents').get()?.count ?? 0,
    ),
    denseVectors: Number(
      database.prepare('SELECT count(*) AS count FROM dense_embeddings').get()?.count ?? 0,
    ),
    projects: Number(
      database.prepare('SELECT count(*) AS count FROM dense_projects').get()?.count ?? 0,
    ),
    projectDocuments: Number(
      database
        .prepare(
          'SELECT count(*) AS count FROM dense_documents WHERE project_key = (SELECT project_key FROM dense_projects WHERE project_identity_digest = ?)',
        )
        .get(PROJECT_IDENTITY_DIGEST)?.count ?? 0,
    ),
  };
  const afterCheckpoint = candidateDatabaseMetrics(database);
  const sqliteVersion = database
    .prepare('SELECT sqlite_version() AS sqlite_version, vec_version() AS vec_version')
    .get();
  database.close();
  sourceState.close();
  sourceCollection.closeSync();
  assertPrototypeStorageGuard();

  const cpu = process.cpuUsage(cpuStarted);
  const ioFinished = processIo();
  const report = readReport();
  report.source = {
    generation: sourcePaths.root.split('/').at(-1),
    manifestVersion: sourceManifest.version,
    sourceCatalogCounts,
    sourceDenseDocuments: counts.denseDocuments,
  };
  report.build = {
    startedAt,
    elapsedSeconds: (performance.now() - started) / 1_000,
    userCpuSeconds: cpu.user / 1_000_000,
    systemCpuSeconds: cpu.system / 1_000_000,
    processReadBytes: ioFinished.readBytes - ioStarted.readBytes,
    processWriteBytes: ioFinished.writeBytes - ioStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStarted,
    peakResidentBytes: processPeakResidentBytes(),
    copiedDocuments,
    counts,
    denseFieldCount: fields.length,
    beforeCheckpoint,
    checkpoint,
    afterCheckpoint,
    sqliteVersion,
    integrity,
    foreignKeyViolations,
  };
  report.limitations = [
    'Linux x64 runtime measured; npm publishes macOS x64 and arm64 binaries, but this run did not execute them.',
    'Best-effort cold-cache measurements use POSIX_FADV_DONTNEED rather than privileged system-wide cache dropping.',
    'The candidate uses released sqlite-vec 0.1.9 exact cosine search and does not test unreleased ANN indexes.',
  ];
  await writeReport(report);
  console.log(JSON.stringify(report.build, null, 2));
}

function queryVectorBlob(embedding: readonly number[]): Uint8Array {
  const values = Float32Array.from(embedding, Math.fround);
  return new Uint8Array(values.buffer);
}

function prepareSqliteDenseQueries(
  database: DatabaseSync,
  vectorTable = 'dense_embeddings',
): {
  global: StatementSync;
  project: StatementSync;
} {
  if (
    vectorTable !== 'dense_embeddings' &&
    vectorTable !== PARTITIONED_VECTOR_TABLE &&
    vectorTable !== BUCKETED_VECTOR_TABLE
  ) {
    throw new Error(`SQLite vec prototype vector table invalid: ${vectorTable}`);
  }
  const table = quoteSqlIdentifier(vectorTable);
  return {
    global: database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM ${table}
        WHERE embedding MATCH ? AND k = ?
      )
      SELECT document.document_id, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `),
    project: database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM ${table}
        WHERE embedding MATCH ? AND k = ? AND project_key = ?
      )
      SELECT document.document_id, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `),
  };
}

function prepareBucketedSqliteDenseQueries(database: DatabaseSync): {
  global: StatementSync;
  project: StatementSync;
} {
  const table = quoteSqlIdentifier(BUCKETED_VECTOR_TABLE);
  return {
    global: database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM ${table}
        WHERE embedding MATCH ? AND k = ?
      )
      SELECT document.document_id, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `),
    project: database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM ${table}
        WHERE embedding MATCH ? AND k = ?
          AND project_bucket = ? AND project_key = ?
      )
      SELECT document.document_id, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `),
  };
}

function queryBucketedSqliteDenseIds(
  statements: ReturnType<typeof prepareBucketedSqliteDenseQueries>,
  vector: Uint8Array,
  projectKey?: number,
): string[] {
  const rows =
    projectKey === undefined
      ? statements.global.all(vector, BigInt(DENSE_SEARCH_LIMIT))
      : statements.project.all(
          vector,
          BigInt(DENSE_SEARCH_LIMIT),
          BigInt(projectKey % PROJECT_BUCKET_COUNT),
          BigInt(projectKey),
        );
  return rows.map((row) => String(row.document_id));
}

function querySqliteDenseIds(
  statements: ReturnType<typeof prepareSqliteDenseQueries>,
  vector: Uint8Array,
  projectKey?: number,
): string[] {
  const rows =
    projectKey === undefined
      ? statements.global.all(vector, BigInt(DENSE_SEARCH_LIMIT))
      : statements.project.all(vector, BigInt(DENSE_SEARCH_LIMIT), BigInt(projectKey));
  return rows.map((row) => String(row.document_id));
}

function querySourceDenseIds(
  source: ZVecCollection,
  embedding: readonly number[],
  projectScoped: boolean,
): string[] {
  return source
    .querySync({
      fieldName: 'embedding',
      vector: embedding.map(Math.fround),
      topk: DENSE_SEARCH_LIMIT,
      outputFields: [],
      includeVector: false,
      ...(projectScoped ? { filter: `projectIdentityDigest = '${PROJECT_IDENTITY_DIGEST}'` } : {}),
    })
    .map((result) => result.id);
}

function bestEffortEvictCandidateFromPageCache(): boolean {
  const script = [
    'import os, sys',
    'fd = os.open(sys.argv[1], os.O_RDONLY)',
    'os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)',
    'os.close(fd)',
  ].join('; ');
  return spawnSync('python', ['-c', script, CANDIDATE_DATABASE_PATH]).status === 0;
}

async function benchmarkCandidate(label: 'beforeChurn' | 'afterChurn'): Promise<void> {
  if (!existsSync(CANDIDATE_DATABASE_PATH)) {
    throw new Error('SQLite vec prototype database missing; run build --reset first');
  }
  const sourcePaths = resolveSourceGeneration();
  const config = await loadRecallConversationConfig();
  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const queryEmbeddings = new Map<string, number[]>();
  for (const query of DENSE_QUERIES) {
    console.error(`Embedding benchmark query: ${query}`);
    queryEmbeddings.set(query, await embeddingProvider.embedQuery(query));
  }

  const firstEmbedding = queryEmbeddings.get(DENSE_QUERIES[0] ?? '');
  if (!firstEmbedding) throw new Error('SQLite vec prototype benchmark query missing');
  const projectKeyDatabase = openCandidateDatabase(false);
  const projectKeyValue = projectKeyDatabase
    .prepare('SELECT project_key FROM dense_projects WHERE project_identity_digest = ?')
    .get(PROJECT_IDENTITY_DIGEST)?.project_key;
  projectKeyDatabase.close();
  if (typeof projectKeyValue !== 'number') {
    throw new Error(`SQLite vec prototype project identity missing: ${PROJECT_IDENTITY}`);
  }

  const coldCache = [];
  for (const scope of ['global', 'project'] as const) {
    const evictionRequested = bestEffortEvictCandidateFromPageCache();
    const coldDatabase = openCandidateDatabase(false);
    const coldStatements = prepareSqliteDenseQueries(coldDatabase);
    const started = performance.now();
    const ids = querySqliteDenseIds(
      coldStatements,
      queryVectorBlob(firstEmbedding),
      scope === 'project' ? projectKeyValue : undefined,
    );
    coldCache.push({
      scope,
      evictionRequested,
      milliseconds: performance.now() - started,
      resultCount: ids.length,
    });
    coldDatabase.close();
  }

  const source = ZVecOpen(sourcePaths.denseCollection, { readOnly: true });
  const database = openCandidateDatabase(false);
  const sqliteStatements = prepareSqliteDenseQueries(database);
  const observations: DenseBenchmarkObservation[] = [];

  for (const scope of ['global', 'project'] as const) {
    for (const query of DENSE_QUERIES) {
      console.error(`Benchmarking ${scope}: ${query}`);
      const embedding = queryEmbeddings.get(query);
      if (!embedding) throw new Error(`SQLite vec prototype embedding missing for ${query}`);
      const vector = queryVectorBlob(embedding);
      const sourceTimes: number[] = [];
      const sqliteTimes: number[] = [];
      let sourceIds: string[] = [];
      let sqliteIds: string[] = [];
      for (let repetition = 0; repetition < 7; repetition += 1) {
        if (repetition % 2 === 0) {
          let started = performance.now();
          sourceIds = querySourceDenseIds(source, embedding, scope === 'project');
          const sourceElapsed = performance.now() - started;
          started = performance.now();
          sqliteIds = querySqliteDenseIds(
            sqliteStatements,
            vector,
            scope === 'project' ? projectKeyValue : undefined,
          );
          const sqliteElapsed = performance.now() - started;
          if (repetition > 0) {
            sourceTimes.push(sourceElapsed);
            sqliteTimes.push(sqliteElapsed);
          }
        } else {
          let started = performance.now();
          sqliteIds = querySqliteDenseIds(
            sqliteStatements,
            vector,
            scope === 'project' ? projectKeyValue : undefined,
          );
          const sqliteElapsed = performance.now() - started;
          started = performance.now();
          sourceIds = querySourceDenseIds(source, embedding, scope === 'project');
          const sourceElapsed = performance.now() - started;
          if (repetition > 0) {
            sourceTimes.push(sourceElapsed);
            sqliteTimes.push(sqliteElapsed);
          }
        }
      }
      observations.push({
        scope,
        query,
        sourceZvecMilliseconds: sourceTimes,
        sqliteVecMilliseconds: sqliteTimes,
        sourceTopIds: sourceIds,
        sqliteTopIds: sqliteIds,
        topResultMatches: sourceIds[0] === sqliteIds[0],
        topEightOverlap: sqliteIds.filter((id) => sourceIds.includes(id)).length,
      });
    }
  }

  const invocationObservations = [];
  const invocationStatement = database.prepare(`
    SELECT invocation.invocation_id
    FROM invocations_fts
    JOIN invocations AS invocation ON invocation.invocation_id = invocations_fts.rowid
    WHERE invocations_fts MATCH ?
    ORDER BY bm25(invocations_fts), invocation.invocation_id
    LIMIT 20
  `);
  for (const query of INVOCATION_QUERIES) {
    const milliseconds: number[] = [];
    let resultCount = 0;
    for (let repetition = 0; repetition < 7; repetition += 1) {
      const started = performance.now();
      resultCount = invocationStatement.all(quoteFtsPhrase(query)).length;
      if (repetition > 0) milliseconds.push(performance.now() - started);
    }
    invocationObservations.push({ query, milliseconds, resultCount });
  }

  database.close();
  source.closeSync();
  const globalObservations = observations.filter((observation) => observation.scope === 'global');
  const projectObservations = observations.filter((observation) => observation.scope === 'project');
  const summarizeScope = (scopeObservations: DenseBenchmarkObservation[]) => {
    const sourceTimes = scopeObservations.flatMap(
      (observation) => observation.sourceZvecMilliseconds,
    );
    const sqliteTimes = scopeObservations.flatMap(
      (observation) => observation.sqliteVecMilliseconds,
    );
    return {
      sourceZvecMedianMilliseconds: percentile(sourceTimes, 0.5),
      sourceZvecP95Milliseconds: percentile(sourceTimes, 0.95),
      sqliteVecMedianMilliseconds: percentile(sqliteTimes, 0.5),
      sqliteVecP95Milliseconds: percentile(sqliteTimes, 0.95),
      matchingTopResults: scopeObservations.filter((observation) => observation.topResultMatches)
        .length,
      minimumTopEightOverlap: Math.min(
        ...scopeObservations.map((observation) => observation.topEightOverlap),
      ),
      queryCount: scopeObservations.length,
    };
  };
  const invocationTimes = invocationObservations.flatMap((observation) => observation.milliseconds);
  const result = {
    measuredAt: new Date().toISOString(),
    projectIdentity: PROJECT_IDENTITY,
    projectDocuments: Number(
      openReadOnlyScalar(
        CANDIDATE_DATABASE_PATH,
        'SELECT count(*) AS count FROM dense_documents WHERE project_key = ?',
        projectKeyValue,
      ),
    ),
    coldCache,
    global: summarizeScope(globalObservations),
    project: summarizeScope(projectObservations),
    observations,
    invocation: {
      medianMilliseconds: percentile(invocationTimes, 0.5),
      p95Milliseconds: percentile(invocationTimes, 0.95),
      observations: invocationObservations,
    },
    peakResidentBytes: processPeakResidentBytes(),
  };
  const report = readReport();
  if (label === 'beforeChurn') report.benchmarkBeforeChurn = result;
  else report.benchmarkAfterChurn = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

function openReadOnlyScalar(path: string, sql: string, ...values: SQLInputValue[]): SQLOutputValue {
  const database = new DatabaseSync(path, { readOnly: true });
  const row = database.prepare(sql).get(...values);
  database.close();
  return row?.count ?? 0;
}

function tableColumns(database: DatabaseSync, table: string): string[] {
  return database
    .prepare(`PRAGMA table_info(${quoteSqlIdentifier(table)})`)
    .all()
    .map((row) => String(row.name));
}

function insertRows(
  database: DatabaseSync,
  table: string,
  columns: readonly string[],
  rows: readonly SqliteRow[],
): void {
  const statement = database.prepare(
    `INSERT INTO ${quoteSqlIdentifier(table)}(${columns.map(quoteSqlIdentifier).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`,
  );
  for (const row of rows) {
    statement.run(...columns.map((column) => row[column] ?? null));
  }
}

function findRepresentativeSessionPath(database: DatabaseSync): string {
  const row = database
    .prepare(`
      SELECT invocation.session_path
      FROM invocations AS invocation
      GROUP BY invocation.session_path
      ORDER BY abs(count(*) - ?), invocation.session_path
      LIMIT 1
    `)
    .get(REPRESENTATIVE_INVOCATION_COUNT);
  if (typeof row?.session_path !== 'string') {
    throw new Error('SQLite vec prototype representative session missing');
  }
  return row.session_path;
}

function loadCandidateSessionData(database: DatabaseSync): CandidateSessionData {
  const sessionPath = findRepresentativeSessionPath(database);
  const sessionDocumentColumns = tableColumns(database, 'session_documents');
  const invocationColumns = tableColumns(database, 'invocations');
  const denseDocumentColumns = tableColumns(database, 'dense_documents');
  const sessionDocuments = database
    .prepare('SELECT * FROM session_documents WHERE session_path = ? ORDER BY document_id')
    .all(sessionPath) as SqliteRow[];
  const invocations = database
    .prepare('SELECT * FROM invocations WHERE session_path = ? ORDER BY invocation_id')
    .all(sessionPath) as SqliteRow[];
  const denseDocumentsToRewrite = database
    .prepare(
      `SELECT * FROM dense_documents WHERE ${quoteSqlIdentifier('sessionPath')} = ? ORDER BY rowid LIMIT 6`,
    )
    .all(sessionPath) as SqliteRow[];
  const rowids = denseDocumentsToRewrite.map((row) => Number(row.rowid));
  const placeholders = rowids.map(() => '?').join(', ');
  const denseVectorsToRewrite =
    rowids.length === 0
      ? []
      : (database
          .prepare(
            `SELECT rowid, embedding, project_key FROM dense_embeddings WHERE rowid IN (${placeholders}) ORDER BY rowid`,
          )
          .all(...rowids) as SqliteRow[]);
  if (denseVectorsToRewrite.length !== denseDocumentsToRewrite.length) {
    throw new Error('SQLite vec prototype representative dense vectors incomplete');
  }
  return {
    sessionPath,
    sessionDocumentColumns,
    sessionDocuments,
    invocationColumns,
    invocations,
    denseDocumentColumns,
    denseDocumentsToRewrite,
    denseVectorsToRewrite,
  };
}

function candidateVectorTables(database: DatabaseSync): string[] {
  return ['dense_embeddings', PARTITIONED_VECTOR_TABLE, BUCKETED_VECTOR_TABLE].filter(
    (table) =>
      Number(
        database.prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?').get(table)
          ?.count ?? 0,
      ) > 0,
  );
}

function replaceCandidateSession(database: DatabaseSync, data: CandidateSessionData): void {
  const vectorTables = candidateVectorTables(database);
  database.exec('BEGIN IMMEDIATE;');
  try {
    database.prepare('DELETE FROM session_documents WHERE session_path = ?').run(data.sessionPath);
    insertRows(database, 'session_documents', data.sessionDocumentColumns, data.sessionDocuments);
    database.prepare('DELETE FROM invocations WHERE session_path = ?').run(data.sessionPath);
    insertRows(database, 'invocations', data.invocationColumns, data.invocations);

    const deleteDocument = database.prepare('DELETE FROM dense_documents WHERE rowid = ?');
    const vectorDeletes = vectorTables.map((table) =>
      database.prepare(`DELETE FROM ${quoteSqlIdentifier(table)} WHERE rowid = ?`),
    );
    for (const document of data.denseDocumentsToRewrite) {
      const rowid = Number(document.rowid);
      for (const deleteVector of vectorDeletes) deleteVector.run(BigInt(rowid));
      deleteDocument.run(rowid);
    }
    insertRows(
      database,
      'dense_documents',
      data.denseDocumentColumns,
      data.denseDocumentsToRewrite,
    );
    const vectorInserts = new Map<string, StatementSync>();
    for (const table of vectorTables) {
      const columns =
        table === BUCKETED_VECTOR_TABLE
          ? 'rowid, embedding, project_bucket, project_key'
          : 'rowid, embedding, project_key';
      const placeholders = table === BUCKETED_VECTOR_TABLE ? '?, ?, ?, ?' : '?, ?, ?';
      vectorInserts.set(
        table,
        database.prepare(
          `INSERT INTO ${quoteSqlIdentifier(table)}(${columns}) VALUES (${placeholders})`,
        ),
      );
    }
    for (const vector of data.denseVectorsToRewrite) {
      const rowid = BigInt(Number(vector.rowid));
      const projectKey = BigInt(Number(vector.project_key));
      for (const [table, insertVector] of vectorInserts) {
        if (table === BUCKETED_VECTOR_TABLE) {
          insertVector.run(
            rowid,
            vector.embedding ?? null,
            projectKey % BigInt(PROJECT_BUCKET_COUNT),
            projectKey,
          );
        } else {
          insertVector.run(rowid, vector.embedding ?? null, projectKey);
        }
      }
    }
    database
      .prepare('UPDATE physical_sessions SET mtime_ms = mtime_ms + 0.001 WHERE session_path = ?')
      .run(data.sessionPath);
    database.exec('COMMIT;');
  } catch (error) {
    database.exec('ROLLBACK;');
    throw error;
  }
}

function sessionSnapshot(database: DatabaseSync, sessionPath: string): Record<string, unknown> {
  const session = database
    .prepare('SELECT * FROM physical_sessions WHERE session_path = ?')
    .get(sessionPath);
  const sessionDocuments = database
    .prepare('SELECT * FROM session_documents WHERE session_path = ? ORDER BY document_id')
    .all(sessionPath);
  const invocations = database
    .prepare('SELECT * FROM invocations WHERE session_path = ? ORDER BY invocation_id')
    .all(sessionPath);
  const denseDocuments = database
    .prepare(
      `SELECT rowid, document_id, checksum FROM dense_documents WHERE ${quoteSqlIdentifier('sessionPath')} = ? ORDER BY rowid`,
    )
    .all(sessionPath);
  const vectorCounts = Object.fromEntries(
    candidateVectorTables(database).map((table) => [
      table,
      Number(
        database
          .prepare(
            `SELECT count(*) AS count FROM ${quoteSqlIdentifier(table)} WHERE rowid IN (SELECT rowid FROM dense_documents WHERE ${quoteSqlIdentifier('sessionPath')} = ?)`,
          )
          .get(sessionPath)?.count ?? 0,
      ),
    ]),
  );
  const hash = createHash('sha256')
    .update(
      JSON.stringify({ session, sessionDocuments, invocations, denseDocuments, vectorCounts }),
    )
    .digest('hex');
  return {
    hash,
    sessionDocuments: sessionDocuments.length,
    invocations: invocations.length,
    denseDocuments: denseDocuments.length,
    vectorCounts,
  };
}

function deleteCandidateSessionInsideOpenTransaction(
  database: DatabaseSync,
  sessionPath: string,
): void {
  database.exec('BEGIN IMMEDIATE;');
  for (const table of candidateVectorTables(database)) {
    database
      .prepare(
        `DELETE FROM ${quoteSqlIdentifier(table)} WHERE rowid IN (SELECT rowid FROM dense_documents WHERE ${quoteSqlIdentifier('sessionPath')} = ?)`,
      )
      .run(sessionPath);
  }
  database
    .prepare(`DELETE FROM dense_documents WHERE ${quoteSqlIdentifier('sessionPath')} = ?`)
    .run(sessionPath);
  database.prepare('DELETE FROM physical_sessions WHERE session_path = ?').run(sessionPath);
}

async function certifyCandidateAtomicity(): Promise<void> {
  const writer = openCandidateDatabase(true);
  const data = loadCandidateSessionData(writer);
  const before = sessionSnapshot(writer, data.sessionPath);

  deleteCandidateSessionInsideOpenTransaction(writer, data.sessionPath);
  const reader = openCandidateDatabase(false);
  const readerDuringWriterTransaction = sessionSnapshot(reader, data.sessionPath);
  reader.close();
  writer.exec('ROLLBACK;');
  const afterExplicitRollback = sessionSnapshot(writer, data.sessionPath);
  writer.close();

  rmSync(CRASH_MARKER_PATH, { force: true });
  const child = spawnSync(
    process.execPath,
    ['--import', 'tsx', import.meta.filename, 'crash-child', data.sessionPath],
    { cwd: dirname(import.meta.filename) },
  );
  const crashMutationReached = existsSync(CRASH_MARKER_PATH);
  rmSync(CRASH_MARKER_PATH, { force: true });

  const recovered = openCandidateDatabase(true);
  const afterCrashRecovery = sessionSnapshot(recovered, data.sessionPath);
  const integrity = String(recovered.prepare('PRAGMA integrity_check').get()?.integrity_check);
  const foreignKeyViolations = recovered.prepare('PRAGMA foreign_key_check').all().length;
  recovered.close();

  const result = {
    representativeSession: data.sessionPath.split('/').at(-1),
    before,
    readerDuringWriterTransaction,
    afterExplicitRollback,
    crashChildSignal: child.signal,
    crashMutationReached,
    afterCrashRecovery,
    readerSawCommittedState: readerDuringWriterTransaction.hash === before.hash,
    explicitRollbackRestoredState: afterExplicitRollback.hash === before.hash,
    crashRecoveryRestoredState: afterCrashRecovery.hash === before.hash,
    integrity,
    foreignKeyViolations,
  };
  const report = readReport();
  report.atomicity = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

function runCrashChild(sessionPath: string): never {
  const database = openCandidateDatabase(true);
  deleteCandidateSessionInsideOpenTransaction(database, sessionPath);
  writeFileSync(CRASH_MARKER_PATH, 'mutated-before-sigkill\n');
  process.kill(process.pid, 'SIGKILL');
  throw new Error('SQLite vec prototype crash child survived SIGKILL');
}

async function benchmarkChangedSessionUpdate(): Promise<void> {
  const database = openCandidateDatabase(true);
  database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const data = loadCandidateSessionData(database);
  const vectorTablesUpdated = candidateVectorTables(database);
  const beforeSnapshot = sessionSnapshot(database, data.sessionPath);
  const beforeStorage = candidateDatabaseMetrics(database);
  const started = performance.now();
  const ioStarted = processIo();
  const deviceStarted = deviceWrittenBytes();
  replaceCandidateSession(database, data);
  database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const ioFinished = processIo();
  const afterSnapshot = sessionSnapshot(database, data.sessionPath);
  const afterStorage = candidateDatabaseMetrics(database);
  const result = {
    representativeSession: data.sessionPath.split('/').at(-1),
    replacedSessionDocuments: data.sessionDocuments.length,
    replacedInvocations: data.invocations.length,
    rewrittenDenseDocumentsAndVectors: data.denseDocumentsToRewrite.length,
    vectorTablesUpdated,
    elapsedSeconds: (performance.now() - started) / 1_000,
    processReadBytes: ioFinished.readBytes - ioStarted.readBytes,
    processWriteBytes: ioFinished.writeBytes - ioStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStarted,
    beforeStorage,
    afterStorage,
    rowCountsPreserved: {
      sessionDocuments: beforeSnapshot.sessionDocuments === afterSnapshot.sessionDocuments,
      invocations: beforeSnapshot.invocations === afterSnapshot.invocations,
      denseDocuments: beforeSnapshot.denseDocuments === afterSnapshot.denseDocuments,
      vectorCounts:
        JSON.stringify(beforeSnapshot.vectorCounts) === JSON.stringify(afterSnapshot.vectorCounts),
    },
  };
  database.close();
  const report = readReport();
  report.changedSessionUpdate = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function benchmarkCandidateChurn(cycles: number): Promise<void> {
  if (!Number.isInteger(cycles) || cycles < 1 || cycles > 1_000) {
    throw new Error('SQLite vec prototype churn cycles must be an integer from 1 to 1000');
  }
  const initial = openCandidateDatabase(true);
  initial.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const data = loadCandidateSessionData(initial);
  const vectorTablesUpdated = candidateVectorTables(initial);
  const beforeStorage = candidateDatabaseMetrics(initial);
  initial.close();

  const started = performance.now();
  const ioStarted = processIo();
  const deviceStarted = deviceWrittenBytes();
  for (let cycle = 1; cycle <= cycles; cycle += 1) {
    const database = openCandidateDatabase(true);
    replaceCandidateSession(database, data);
    database.close();
    if (cycle % 10 === 0) {
      assertPrototypeStorageGuard();
      console.error(`  completed ${cycle} / ${cycles} changed-session cycles`);
    }
  }
  const final = openCandidateDatabase(true);
  final.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const afterStorage = candidateDatabaseMetrics(final);
  const integrity = String(final.prepare('PRAGMA integrity_check').get()?.integrity_check);
  const foreignKeyViolations = final.prepare('PRAGMA foreign_key_check').all().length;
  const finalSnapshot = sessionSnapshot(final, data.sessionPath);
  final.close();
  const ioFinished = processIo();
  const elapsedSeconds = (performance.now() - started) / 1_000;
  const result = {
    cycles,
    vectorTablesUpdated,
    elapsedSeconds,
    averageMillisecondsPerCycle: (elapsedSeconds * 1_000) / cycles,
    processReadBytes: ioFinished.readBytes - ioStarted.readBytes,
    processWriteBytes: ioFinished.writeBytes - ioStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStarted,
    averageDeviceWrittenBytesPerCycle: (deviceWrittenBytes() - deviceStarted) / cycles,
    beforeStorage,
    afterStorage,
    allocatedGrowthBytes:
      Number(afterStorage.allocatedBytes ?? 0) - Number(beforeStorage.allocatedBytes ?? 0),
    freePageGrowth: Number(afterStorage.freePages ?? 0) - Number(beforeStorage.freePages ?? 0),
    finalSnapshot,
    integrity,
    foreignKeyViolations,
  };
  const report = readReport();
  report.churn = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function buildPartitionedVectorAlternative(reset: boolean): Promise<void> {
  const database = openCandidateDatabase(true);
  const tableExists =
    Number(
      database
        .prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?')
        .get(PARTITIONED_VECTOR_TABLE)?.count ?? 0,
    ) > 0;
  if (tableExists) {
    if (!reset) {
      database.close();
      throw new Error(
        `SQLite vec partitioned alternative already exists; pass --reset to replace it`,
      );
    }
    database.exec(`DROP TABLE ${quoteSqlIdentifier(PARTITIONED_VECTOR_TABLE)};`);
    database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  }

  const beforeStorage = candidateDatabaseMetrics(database);
  const started = performance.now();
  const ioStarted = processIo();
  const deviceStarted = deviceWrittenBytes();
  database.exec(`
    PRAGMA wal_autocheckpoint = 0;
    CREATE VIRTUAL TABLE ${quoteSqlIdentifier(PARTITIONED_VECTOR_TABLE)} USING vec0(
      embedding FLOAT[1024] DISTANCE_METRIC=cosine,
      project_key INTEGER PARTITION KEY
    );
    BEGIN IMMEDIATE;
  `);
  let transactionOpen = true;
  try {
    const count = Number(
      database.prepare('SELECT count(*) AS count FROM dense_embeddings').get()?.count ?? 0,
    );
    const insertBatch = database.prepare(`
      INSERT INTO ${quoteSqlIdentifier(PARTITIONED_VECTOR_TABLE)}(rowid, embedding, project_key)
      SELECT rowid, embedding, project_key
      FROM dense_embeddings
      WHERE rowid BETWEEN ? AND ?
    `);
    for (let start = 1; start <= count; start += 10_000) {
      const end = Math.min(count, start + 9_999);
      insertBatch.run(BigInt(start), BigInt(end));
      assertPrototypeStorageGuard();
      console.error(`  partitioned ${end.toLocaleString()} / ${count.toLocaleString()} vectors`);
    }
    database.exec('COMMIT;');
    transactionOpen = false;
  } catch (error) {
    if (transactionOpen) database.exec('ROLLBACK;');
    database.close();
    throw error;
  }
  const beforeCheckpoint = candidateDatabaseMetrics(database);
  database.exec('PRAGMA wal_checkpoint(TRUNCATE); PRAGMA wal_autocheckpoint = 1000;');
  const afterStorage = candidateDatabaseMetrics(database);
  const partitionedCount = Number(
    database
      .prepare(`SELECT count(*) AS count FROM ${quoteSqlIdentifier(PARTITIONED_VECTOR_TABLE)}`)
      .get()?.count ?? 0,
  );
  const projectDistribution = database
    .prepare(`
      SELECT min(document_count) AS minimum,
             max(document_count) AS maximum,
             avg(document_count) AS average,
             sum(document_count < 100) AS projects_below_100
      FROM (
        SELECT project_key, count(*) AS document_count
        FROM dense_documents
        GROUP BY project_key
      )
    `)
    .get();
  const ioFinished = processIo();
  const result = {
    elapsedSeconds: (performance.now() - started) / 1_000,
    processReadBytes: ioFinished.readBytes - ioStarted.readBytes,
    processWriteBytes: ioFinished.writeBytes - ioStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStarted,
    partitionedVectors: partitionedCount,
    beforeStorage,
    beforeCheckpoint,
    afterStorage,
    additionalAllocatedBytes:
      Number(afterStorage.allocatedBytes ?? 0) - Number(beforeStorage.allocatedBytes ?? 0),
    projectDistribution,
  };
  database.close();
  assertPrototypeStorageGuard();
  const report = readReport();
  report.partitionedAlternativeBuild = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function benchmarkPartitionedVectorAlternative(): Promise<void> {
  const sourcePaths = resolveSourceGeneration();
  const config = await loadRecallConversationConfig();
  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const database = openCandidateDatabase(false);
  const tableExists =
    Number(
      database
        .prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?')
        .get(PARTITIONED_VECTOR_TABLE)?.count ?? 0,
    ) > 0;
  if (!tableExists) {
    database.close();
    throw new Error('Build the SQLite vec partitioned alternative first');
  }
  const projectKeyValue = database
    .prepare('SELECT project_key FROM dense_projects WHERE project_identity_digest = ?')
    .get(PROJECT_IDENTITY_DIGEST)?.project_key;
  if (typeof projectKeyValue !== 'number') {
    database.close();
    throw new Error(`SQLite vec prototype project identity missing: ${PROJECT_IDENTITY}`);
  }
  const metadataStatements = prepareSqliteDenseQueries(database);
  const partitionedStatements = prepareSqliteDenseQueries(database, PARTITIONED_VECTOR_TABLE);
  const source = ZVecOpen(sourcePaths.denseCollection, { readOnly: true });
  const observations: PartitionedDenseBenchmarkObservation[] = [];

  for (const scope of ['global', 'project'] as const) {
    for (const query of DENSE_QUERIES) {
      console.error(`Benchmarking partition alternative ${scope}: ${query}`);
      const embedding = await embeddingProvider.embedQuery(query);
      const vector = queryVectorBlob(embedding);
      const sourceTimes: number[] = [];
      const metadataTimes: number[] = [];
      const partitionedTimes: number[] = [];
      let sourceIds: string[] = [];
      let metadataIds: string[] = [];
      let partitionedIds: string[] = [];
      for (let repetition = 0; repetition < 7; repetition += 1) {
        let started = performance.now();
        sourceIds = querySourceDenseIds(source, embedding, scope === 'project');
        const sourceElapsed = performance.now() - started;
        started = performance.now();
        metadataIds = querySqliteDenseIds(
          metadataStatements,
          vector,
          scope === 'project' ? projectKeyValue : undefined,
        );
        const metadataElapsed = performance.now() - started;
        started = performance.now();
        partitionedIds = querySqliteDenseIds(
          partitionedStatements,
          vector,
          scope === 'project' ? projectKeyValue : undefined,
        );
        const partitionedElapsed = performance.now() - started;
        if (repetition > 0) {
          sourceTimes.push(sourceElapsed);
          metadataTimes.push(metadataElapsed);
          partitionedTimes.push(partitionedElapsed);
        }
      }
      observations.push({
        scope,
        query,
        sourceZvecMilliseconds: sourceTimes,
        metadataSqliteVecMilliseconds: metadataTimes,
        partitionedSqliteVecMilliseconds: partitionedTimes,
        sourceTopIds: sourceIds,
        metadataTopIds: metadataIds,
        partitionedTopIds: partitionedIds,
        partitionedTopResultMatches: sourceIds[0] === partitionedIds[0],
        partitionedTopEightOverlap: partitionedIds.filter((id) => sourceIds.includes(id)).length,
      });
    }
  }
  source.closeSync();
  database.close();

  const summarize = (scope: 'global' | 'project') => {
    const scoped = observations.filter((observation) => observation.scope === scope);
    const sourceTimes = scoped.flatMap((observation) => observation.sourceZvecMilliseconds);
    const metadataTimes = scoped.flatMap(
      (observation) => observation.metadataSqliteVecMilliseconds,
    );
    const partitionedTimes = scoped.flatMap(
      (observation) => observation.partitionedSqliteVecMilliseconds,
    );
    return {
      sourceZvecMedianMilliseconds: percentile(sourceTimes, 0.5),
      sourceZvecP95Milliseconds: percentile(sourceTimes, 0.95),
      metadataSqliteVecMedianMilliseconds: percentile(metadataTimes, 0.5),
      metadataSqliteVecP95Milliseconds: percentile(metadataTimes, 0.95),
      partitionedSqliteVecMedianMilliseconds: percentile(partitionedTimes, 0.5),
      partitionedSqliteVecP95Milliseconds: percentile(partitionedTimes, 0.95),
      matchingTopResults: scoped.filter((observation) => observation.partitionedTopResultMatches)
        .length,
      minimumTopEightOverlap: Math.min(
        ...scoped.map((observation) => observation.partitionedTopEightOverlap),
      ),
    };
  };
  const result = {
    measuredAt: new Date().toISOString(),
    global: summarize('global'),
    project: summarize('project'),
    observations,
  };
  const report = readReport();
  report.partitionedAlternativeBenchmark = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function buildBucketedVectorAlternative(reset: boolean): Promise<void> {
  const database = openCandidateDatabase(true);
  const bucketedExists =
    Number(
      database
        .prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?')
        .get(BUCKETED_VECTOR_TABLE)?.count ?? 0,
    ) > 0;
  if (bucketedExists && !reset) {
    database.close();
    throw new Error('SQLite vec bucketed alternative exists; pass --reset to replace it');
  }

  const initialStorage = candidateDatabaseMetrics(database);
  database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  if (bucketedExists) {
    database.exec(`DROP TABLE ${quoteSqlIdentifier(BUCKETED_VECTOR_TABLE)};`);
  }
  const partitionedExists =
    Number(
      database
        .prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?')
        .get(PARTITIONED_VECTOR_TABLE)?.count ?? 0,
    ) > 0;
  if (partitionedExists) {
    database.exec(`DROP TABLE ${quoteSqlIdentifier(PARTITIONED_VECTOR_TABLE)};`);
  }
  database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const afterDropStorage = candidateDatabaseMetrics(database);

  const started = performance.now();
  const ioStarted = processIo();
  const deviceStarted = deviceWrittenBytes();
  database.exec(`
    PRAGMA wal_autocheckpoint = 1000;
    CREATE VIRTUAL TABLE ${quoteSqlIdentifier(BUCKETED_VECTOR_TABLE)} USING vec0(
      embedding FLOAT[1024] DISTANCE_METRIC=cosine,
      project_bucket INTEGER PARTITION KEY,
      project_key INTEGER
    );
  `);
  const count = Number(
    database.prepare('SELECT count(*) AS count FROM dense_embeddings').get()?.count ?? 0,
  );
  const insertBatch = database.prepare(`
    INSERT INTO ${quoteSqlIdentifier(BUCKETED_VECTOR_TABLE)}(
      rowid, embedding, project_bucket, project_key
    )
    SELECT rowid, embedding, project_key % ${PROJECT_BUCKET_COUNT}, project_key
    FROM dense_embeddings
    WHERE rowid BETWEEN ? AND ?
  `);
  for (let start = 1; start <= count; start += 10_000) {
    const end = Math.min(count, start + 9_999);
    database.exec('BEGIN IMMEDIATE;');
    try {
      insertBatch.run(BigInt(start), BigInt(end));
      database.exec('COMMIT;');
    } catch (error) {
      database.exec('ROLLBACK;');
      database.close();
      throw error;
    }
    assertPrototypeStorageGuard();
    console.error(`  bucketed ${end.toLocaleString()} / ${count.toLocaleString()} vectors`);
  }
  database.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  const afterStorage = candidateDatabaseMetrics(database);
  const bucketDistribution = database
    .prepare(`
      SELECT min(document_count) AS minimum,
             max(document_count) AS maximum,
             avg(document_count) AS average
      FROM (
        SELECT project_key % ${PROJECT_BUCKET_COUNT} AS bucket, count(*) AS document_count
        FROM dense_documents
        GROUP BY bucket
      )
    `)
    .get();
  const bucketedCount = Number(
    database
      .prepare(`SELECT count(*) AS count FROM ${quoteSqlIdentifier(BUCKETED_VECTOR_TABLE)}`)
      .get()?.count ?? 0,
  );
  const ioFinished = processIo();
  const result = {
    bucketCount: PROJECT_BUCKET_COUNT,
    elapsedSeconds: (performance.now() - started) / 1_000,
    processReadBytes: ioFinished.readBytes - ioStarted.readBytes,
    processWriteBytes: ioFinished.writeBytes - ioStarted.writeBytes,
    deviceWrittenBytes: deviceWrittenBytes() - deviceStarted,
    bucketedVectors: bucketedCount,
    initialStorage,
    afterDropStorage,
    afterStorage,
    bucketDistribution,
    reusedFreeBytes: Number(afterDropStorage.freeBytes ?? 0) - Number(afterStorage.freeBytes ?? 0),
  };
  database.close();
  const report = readReport();
  report.bucketedAlternativeBuild = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function benchmarkBucketedVectorAlternative(): Promise<void> {
  const sourcePaths = resolveSourceGeneration();
  const config = await loadRecallConversationConfig();
  const embeddingProvider = createOctenHttpEmbeddingProvider({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const database = openCandidateDatabase(false);
  const bucketedExists =
    Number(
      database
        .prepare('SELECT count(*) AS count FROM sqlite_master WHERE name = ?')
        .get(BUCKETED_VECTOR_TABLE)?.count ?? 0,
    ) > 0;
  if (!bucketedExists) {
    database.close();
    throw new Error('Build the SQLite vec bucketed alternative first');
  }
  const projectKeyValue = database
    .prepare('SELECT project_key FROM dense_projects WHERE project_identity_digest = ?')
    .get(PROJECT_IDENTITY_DIGEST)?.project_key;
  if (typeof projectKeyValue !== 'number') {
    database.close();
    throw new Error(`SQLite vec prototype project identity missing: ${PROJECT_IDENTITY}`);
  }
  const metadataStatements = prepareSqliteDenseQueries(database);
  const bucketedStatements = prepareBucketedSqliteDenseQueries(database);
  const source = ZVecOpen(sourcePaths.denseCollection, { readOnly: true });
  const observations: BucketedDenseBenchmarkObservation[] = [];

  for (const scope of ['global', 'project'] as const) {
    for (const query of DENSE_QUERIES) {
      console.error(`Benchmarking bucket alternative ${scope}: ${query}`);
      const embedding = await embeddingProvider.embedQuery(query);
      const vector = queryVectorBlob(embedding);
      const sourceTimes: number[] = [];
      const metadataTimes: number[] = [];
      const bucketedTimes: number[] = [];
      let sourceIds: string[] = [];
      let metadataIds: string[] = [];
      let bucketedIds: string[] = [];
      for (let repetition = 0; repetition < 7; repetition += 1) {
        let started = performance.now();
        sourceIds = querySourceDenseIds(source, embedding, scope === 'project');
        const sourceElapsed = performance.now() - started;
        started = performance.now();
        metadataIds = querySqliteDenseIds(
          metadataStatements,
          vector,
          scope === 'project' ? projectKeyValue : undefined,
        );
        const metadataElapsed = performance.now() - started;
        started = performance.now();
        bucketedIds = queryBucketedSqliteDenseIds(
          bucketedStatements,
          vector,
          scope === 'project' ? projectKeyValue : undefined,
        );
        const bucketedElapsed = performance.now() - started;
        if (repetition > 0) {
          sourceTimes.push(sourceElapsed);
          metadataTimes.push(metadataElapsed);
          bucketedTimes.push(bucketedElapsed);
        }
      }
      observations.push({
        scope,
        query,
        sourceZvecMilliseconds: sourceTimes,
        metadataSqliteVecMilliseconds: metadataTimes,
        bucketedSqliteVecMilliseconds: bucketedTimes,
        sourceTopIds: sourceIds,
        metadataTopIds: metadataIds,
        bucketedTopIds: bucketedIds,
        bucketedTopResultMatches: sourceIds[0] === bucketedIds[0],
        bucketedTopEightOverlap: bucketedIds.filter((id) => sourceIds.includes(id)).length,
      });
    }
  }
  source.closeSync();
  database.close();

  const summarize = (scope: 'global' | 'project') => {
    const scoped = observations.filter((observation) => observation.scope === scope);
    const sourceTimes = scoped.flatMap((observation) => observation.sourceZvecMilliseconds);
    const metadataTimes = scoped.flatMap(
      (observation) => observation.metadataSqliteVecMilliseconds,
    );
    const bucketedTimes = scoped.flatMap(
      (observation) => observation.bucketedSqliteVecMilliseconds,
    );
    return {
      sourceZvecMedianMilliseconds: percentile(sourceTimes, 0.5),
      sourceZvecP95Milliseconds: percentile(sourceTimes, 0.95),
      metadataSqliteVecMedianMilliseconds: percentile(metadataTimes, 0.5),
      metadataSqliteVecP95Milliseconds: percentile(metadataTimes, 0.95),
      bucketedSqliteVecMedianMilliseconds: percentile(bucketedTimes, 0.5),
      bucketedSqliteVecP95Milliseconds: percentile(bucketedTimes, 0.95),
      matchingTopResults: scoped.filter((observation) => observation.bucketedTopResultMatches)
        .length,
      minimumTopEightOverlap: Math.min(
        ...scoped.map((observation) => observation.bucketedTopEightOverlap),
      ),
    };
  };
  const result = {
    measuredAt: new Date().toISOString(),
    bucketCount: PROJECT_BUCKET_COUNT,
    global: summarize('global'),
    project: summarize('project'),
    observations,
  };
  const report = readReport();
  report.bucketedAlternativeBenchmark = result;
  await writeReport(report);
  console.log(JSON.stringify(result, null, 2));
}

async function runAll(reset: boolean): Promise<void> {
  await buildSqliteVecCandidate(reset);
  await benchmarkCandidate('beforeChurn');
  await certifyCandidateAtomicity();
  await benchmarkChangedSessionUpdate();
  await benchmarkCandidateChurn(DEFAULT_CHURN_CYCLES);
  await benchmarkCandidate('afterChurn');
  await buildPartitionedVectorAlternative(false);
  await benchmarkPartitionedVectorAlternative();
  await buildBucketedVectorAlternative(false);
  await benchmarkBucketedVectorAlternative();
}

async function main(): Promise<void> {
  const [mode, ...arguments_] = process.argv.slice(2);
  const reset = arguments_.includes('--reset');
  switch (mode) {
    case 'build':
      await buildSqliteVecCandidate(reset);
      return;
    case 'benchmark-before-churn':
      await benchmarkCandidate('beforeChurn');
      return;
    case 'atomicity':
      await certifyCandidateAtomicity();
      return;
    case 'update':
      await benchmarkChangedSessionUpdate();
      return;
    case 'churn': {
      const cyclesArgument = arguments_.find((argument) => argument.startsWith('--cycles='));
      const cycles = cyclesArgument
        ? Number(cyclesArgument.slice('--cycles='.length))
        : DEFAULT_CHURN_CYCLES;
      await benchmarkCandidateChurn(cycles);
      return;
    }
    case 'benchmark-after-churn':
      await benchmarkCandidate('afterChurn');
      return;
    case 'build-partitioned':
      await buildPartitionedVectorAlternative(reset);
      return;
    case 'benchmark-partitioned':
      await benchmarkPartitionedVectorAlternative();
      return;
    case 'build-bucketed':
      await buildBucketedVectorAlternative(reset);
      return;
    case 'benchmark-bucketed':
      await benchmarkBucketedVectorAlternative();
      return;
    case 'crash-child': {
      const sessionPath = arguments_[0];
      if (!sessionPath) throw new Error('SQLite vec prototype crash child session path missing');
      runCrashChild(sessionPath);
    }
    case 'all':
      await runAll(reset);
      return;
    default:
      throw new Error(
        'Usage: npm run prototype:sqlite-vec -- <build|benchmark-before-churn|atomicity|update|churn|benchmark-after-churn|build-partitioned|benchmark-partitioned|build-bucketed|benchmark-bucketed|all> [--reset] [--cycles=N]',
      );
  }
}

await main();
