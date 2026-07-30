import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, realpath, rename, rm, stat } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  RecallBacklogFailureCategory,
  RecallGenerationCutoverState,
  RECALL_INDEX_MANIFEST_VERSION,
} from './enums.js';
import { RecallGenerationPointerError } from './errors.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import { syncRecallDirectory } from './sync-recall-directory.js';
import { RECALL_WORK_MARKER_VERSION } from './recall-work-marker.js';

/** Current strict version for the checksummed active-generation pointer. */
export const RECALL_ACTIVE_GENERATION_POINTER_VERSION = 1;

/** Current strict version for the generation build and cutover registry. */
export const RECALL_GENERATION_REGISTRY_VERSION = 1;

/** Current strict version for the scalar material recall backlog summary. */
export const RECALL_BACKLOG_SUMMARY_VERSION = 1;

/** Service objective starts after the longest expected 30-minute quiescence window. */
export const RECALL_MATERIAL_BACKLOG_SERVICE_OBJECTIVE_MILLISECONDS = 30 * 60_000;

const RECALL_GENERATION_IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]+$/u;

/** Pointer-selected immutable paths used by one read-only recall search. */
export interface RecallActiveGenerationSelection {
  activeGenerationId: string;
  generationDirectory: string;
  databasePath: string;
  projectionDatabasePath: string;
  statePath: string;
  manifestPath: string;
}

/** Minimal atomic pointer selecting the only generation available to search. */
export interface RecallActiveGenerationPointer {
  version: 1;
  activeGenerationId: string;
  checksum: string;
}

/** Checksum input for one versioned active-generation pointer. */
export interface RecallActiveGenerationPointerIdentity {
  version: 1;
  activeGenerationId: string;
}

/** One generation's durable explicit-rebuild, replay, rollback, and retention state. */
export interface RecallGenerationRegistryEntry {
  generationId: string;
  state: RecallGenerationCutoverState;
  /** Configured embedding semantics used to build this generation, when known. */
  embeddingProfileId?: string;
  indexManifestVersion: 5 | 6;
  markerSchemaVersion: 1 | null;
  sessionProjectionSchemaVersion: 3 | null;
  indexManifestFingerprint: string;
  rebuildStartedAtEpochMilliseconds: number;
  stateChangedAtEpochMilliseconds: number;
  rebuildStartMarkerId: string | null;
  rebuildMarkerWatermark?: string[];
  /** Immutable generation-local snapshot file governing the current fixed replay. */
  replaySnapshotFileName?: string;
  validatedAtEpochMilliseconds?: number | null;
  retireAfterEpochMilliseconds?: number | null;
}

/** Durable registry; active selection is nullable only before the first successful cutover. */
export interface RecallGenerationRegistry {
  version: 1;
  activeGenerationId: string | null;
  buildingGenerationId: string | null;
  rollbackGenerationId: string | null;
  activePointerChecksum: string | null;
  generations: RecallGenerationRegistryEntry[];
}

/** Scalar-only material backlog state safe to read and display during search. */
export interface RecallBacklogSummary {
  version: 1;
  pendingEligibleSessionCount: number;
  oldestEligibleMarkerAgeMilliseconds: number | null;
  activeGenerationId: string;
  buildingGenerationId: string | null;
  generationState: RecallGenerationCutoverState;
  activeGenerationAgeMilliseconds: number;
  rebuildAgeMilliseconds: number | null;
  lastFailureCategory: RecallBacklogFailureCategory | null;
  observedAtEpochMilliseconds: number;
}

/** Writable file capability used by atomic generation-state fault tests. */
export interface RecallGenerationStateFile {
  writeFile(content: string): Promise<void>;
  sync(): Promise<void>;
  close(): Promise<void>;
}

/** Minimal filesystem boundary for checksummed atomic generation-state replacement. */
export interface RecallGenerationStateFilesystem {
  createDirectory(path: string): Promise<void>;
  openExclusiveFile(path: string): Promise<RecallGenerationStateFile>;
  renameFile(temporaryPath: string, destinationPath: string): Promise<void>;
  syncDirectory(path: string): Promise<void>;
  removeFile(path: string): Promise<void>;
}

/** Optional fault-injection boundary for durable generation-state writes. */
export interface RecallGenerationStateWriteOptions {
  filesystem?: RecallGenerationStateFilesystem;
}

const nodeGenerationStateFilesystem: RecallGenerationStateFilesystem = {
  async createDirectory(path) {
    await mkdir(path, { recursive: true });
  },
  async openExclusiveFile(path) {
    return open(path, 'wx', 0o600);
  },
  renameFile: rename,
  syncDirectory: syncRecallDirectory,
  async removeFile(path) {
    await rm(path, { force: true });
  },
};

const nonemptyStringSchema = Type.String({ minLength: 1 });
const generationIdentifierSchema = Type.String({ pattern: '^[A-Za-z0-9_-]+$' });
const nullableIdentifierSchema = Type.Union([generationIdentifierSchema, Type.Null()]);
const nullableChecksumSchema = Type.Union([
  Type.String({ pattern: '^[a-f0-9]{64}$' }),
  Type.Null(),
]);
const recallActiveGenerationPointerSchema = Type.Object(
  {
    version: Type.Literal(RECALL_ACTIVE_GENERATION_POINTER_VERSION),
    activeGenerationId: generationIdentifierSchema,
    checksum: Type.String({ pattern: '^[a-f0-9]{64}$' }),
  },
  { additionalProperties: false },
);
const recallGenerationRegistryEntrySchema = Type.Object(
  {
    generationId: generationIdentifierSchema,
    state: Type.Enum(RecallGenerationCutoverState),
    embeddingProfileId: Type.Optional(nonemptyStringSchema),
    indexManifestVersion: Type.Union([
      Type.Literal(5),
      Type.Literal(RECALL_INDEX_MANIFEST_VERSION),
    ]),
    markerSchemaVersion: Type.Union([Type.Literal(RECALL_WORK_MARKER_VERSION), Type.Null()]),
    sessionProjectionSchemaVersion: Type.Union([
      Type.Literal(RECALL_SESSION_PROJECTION_SCHEMA_VERSION),
      Type.Null(),
    ]),
    indexManifestFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    rebuildStartedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    stateChangedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    rebuildStartMarkerId: Type.Union([generationIdentifierSchema, Type.Null()]),
    rebuildMarkerWatermark: Type.Optional(Type.Array(generationIdentifierSchema)),
    replaySnapshotFileName: Type.Optional(
      Type.String({ pattern: '^generation-replay-snapshot(?:-[A-Za-z0-9_-]+)?\\.json$' }),
    ),
    validatedAtEpochMilliseconds: Type.Optional(
      Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
    ),
    retireAfterEpochMilliseconds: Type.Optional(
      Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
    ),
  },
  { additionalProperties: false },
);
const recallGenerationRegistrySchema = Type.Object(
  {
    version: Type.Literal(RECALL_GENERATION_REGISTRY_VERSION),
    activeGenerationId: nullableIdentifierSchema,
    buildingGenerationId: nullableIdentifierSchema,
    rollbackGenerationId: nullableIdentifierSchema,
    activePointerChecksum: nullableChecksumSchema,
    generations: Type.Array(recallGenerationRegistryEntrySchema),
  },
  { additionalProperties: false },
);
const recallBacklogSummarySchema = Type.Object(
  {
    version: Type.Literal(RECALL_BACKLOG_SUMMARY_VERSION),
    pendingEligibleSessionCount: Type.Integer({ minimum: 0 }),
    oldestEligibleMarkerAgeMilliseconds: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
    activeGenerationId: nonemptyStringSchema,
    buildingGenerationId: nullableIdentifierSchema,
    generationState: Type.Enum(RecallGenerationCutoverState),
    activeGenerationAgeMilliseconds: Type.Integer({ minimum: 0 }),
    rebuildAgeMilliseconds: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
    lastFailureCategory: Type.Union([Type.Enum(RecallBacklogFailureCategory), Type.Null()]),
    observedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

function parseJsonContract(source: string, contractName: string): unknown {
  try {
    return JSON.parse(source);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`${contractName} unreadable: ${message}`, { cause: error });
  }
}

function parseActiveGenerationPointer(value: unknown): RecallActiveGenerationPointer {
  let pointer: RecallActiveGenerationPointer;
  try {
    pointer = Value.Parse(recallActiveGenerationPointerSchema, value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall active generation pointer invalid: ${message}`, { cause: error });
  }
  const expectedChecksum = calculateRecallActiveGenerationPointerChecksum(pointer);
  if (pointer.checksum !== expectedChecksum) {
    throw new Error(
      `Recall active generation pointer checksum mismatch: expected ${expectedChecksum}, received ${pointer.checksum}`,
    );
  }
  return pointer;
}

/** Calculates the canonical SHA-256 checksum covering pointer version and generation ID. */
export function calculateRecallActiveGenerationPointerChecksum(
  pointer: RecallActiveGenerationPointerIdentity,
): string {
  return createHash('sha256')
    .update(
      JSON.stringify({ version: pointer.version, activeGenerationId: pointer.activeGenerationId }),
    )
    .digest('hex');
}

/** Creates one complete checksummed active-generation pointer without writing it. */
export function createRecallActiveGenerationPointer(
  activeGenerationId: string,
): RecallActiveGenerationPointer {
  if (!RECALL_GENERATION_IDENTIFIER_PATTERN.test(activeGenerationId)) {
    throw new Error('Recall active generation pointer generation ID invalid');
  }
  const identity: RecallActiveGenerationPointerIdentity = {
    version: RECALL_ACTIVE_GENERATION_POINTER_VERSION,
    activeGenerationId,
  };
  return { ...identity, checksum: calculateRecallActiveGenerationPointerChecksum(identity) };
}

/** Strictly validates and serializes one active-generation pointer without writing it. */
export function encodeRecallActiveGenerationPointer(
  pointer: RecallActiveGenerationPointer,
): string {
  return `${JSON.stringify(parseActiveGenerationPointer(pointer))}\n`;
}

/** Strictly parses and checksum-verifies one active-generation pointer. */
export function decodeRecallActiveGenerationPointer(source: string): RecallActiveGenerationPointer {
  return parseActiveGenerationPointer(
    parseJsonContract(source, 'Recall active generation pointer'),
  );
}

function findRegistryGeneration(
  registry: RecallGenerationRegistry,
  generationId: string,
  role: string,
): RecallGenerationRegistryEntry {
  const generation = registry.generations.find(
    (candidate) => candidate.generationId === generationId,
  );
  if (!generation) {
    throw new Error(
      `Recall generation registry ${role} generation missing from entries: ${generationId}`,
    );
  }
  return generation;
}

function assertUniqueGenerationValues(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`Recall generation registry contains duplicate ${label}`);
  }
}

function assertRecallGenerationRegistryInvariants(registry: RecallGenerationRegistry): void {
  assertUniqueGenerationValues(
    registry.generations.map(({ generationId }) => generationId),
    'generation IDs',
  );
  for (const entry of registry.generations) {
    assertUniqueGenerationValues(
      entry.rebuildMarkerWatermark ?? [],
      'rebuild marker watermark IDs',
    );
    if (entry.indexManifestVersion === 5) {
      if (entry.markerSchemaVersion !== null || entry.sessionProjectionSchemaVersion !== null) {
        throw new Error(
          'Recall legacy generation registry entry must not synthesize incremental schemas',
        );
      }
      if (
        entry.state !== RecallGenerationCutoverState.LEGACY_READ_ONLY &&
        entry.state !== RecallGenerationCutoverState.ROLLBACK &&
        entry.state !== RecallGenerationCutoverState.RETIRED
      ) {
        throw new Error('Recall version-5 generation registry entry must remain read-only');
      }
    }
  }
  if (registry.activeGenerationId === null) {
    if (registry.activePointerChecksum !== null || registry.rollbackGenerationId !== null) {
      throw new Error(
        'Recall generation registry bootstrap state cannot reference active or rollback state',
      );
    }
  } else {
    const expectedPointer = createRecallActiveGenerationPointer(registry.activeGenerationId);
    if (registry.activePointerChecksum !== expectedPointer.checksum) {
      throw new Error(
        `Recall generation registry active pointer checksum mismatch: expected ${expectedPointer.checksum}, received ${registry.activePointerChecksum}`,
      );
    }
    const active = findRegistryGeneration(registry, registry.activeGenerationId, 'active');
    if (
      active.state !== RecallGenerationCutoverState.ACTIVE &&
      active.state !== RecallGenerationCutoverState.REPLAY_PENDING &&
      active.state !== RecallGenerationCutoverState.LEGACY_READ_ONLY
    ) {
      throw new Error(
        `Recall generation registry active generation has invalid state: ${active.state}`,
      );
    }
  }
  if (registry.buildingGenerationId !== null) {
    const building = findRegistryGeneration(registry, registry.buildingGenerationId, 'building');
    if (
      building.state !== RecallGenerationCutoverState.BUILDING &&
      building.state !== RecallGenerationCutoverState.READY
    ) {
      throw new Error(
        `Recall generation registry building generation has invalid state: ${building.state}`,
      );
    }
  }
  if (registry.rollbackGenerationId !== null) {
    const rollback = findRegistryGeneration(registry, registry.rollbackGenerationId, 'rollback');
    if (rollback.state !== RecallGenerationCutoverState.ROLLBACK) {
      throw new Error(
        `Recall generation registry rollback generation has invalid state: ${rollback.state}`,
      );
    }
  }
}

function parseRecallGenerationRegistry(value: unknown): RecallGenerationRegistry {
  let registry: RecallGenerationRegistry;
  try {
    registry = Value.Parse(recallGenerationRegistrySchema, value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall generation registry invalid: ${message}`, { cause: error });
  }
  assertRecallGenerationRegistryInvariants(registry);
  return registry;
}

/** Strictly validates and serializes one generation registry without writing it. */
export function encodeRecallGenerationRegistry(registry: RecallGenerationRegistry): string {
  return `${JSON.stringify(parseRecallGenerationRegistry(registry))}\n`;
}

/** Strictly parses one generation registry and verifies pointer and role consistency. */
export function decodeRecallGenerationRegistry(source: string): RecallGenerationRegistry {
  return parseRecallGenerationRegistry(parseJsonContract(source, 'Recall generation registry'));
}

function parseRecallBacklogSummary(value: unknown): RecallBacklogSummary {
  try {
    return Value.Parse(recallBacklogSummarySchema, value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall backlog summary invalid: ${message}`, { cause: error });
  }
}

/** Strictly validates and serializes one scalar-only material backlog summary. */
export function encodeRecallBacklogSummary(summary: RecallBacklogSummary): string {
  return `${JSON.stringify(parseRecallBacklogSummary(summary))}\n`;
}

/** Strictly parses one scalar-only material backlog summary. */
export function decodeRecallBacklogSummary(source: string): RecallBacklogSummary {
  return parseRecallBacklogSummary(parseJsonContract(source, 'Recall backlog summary'));
}

async function writeAtomicRecallGenerationState(
  destinationPath: string,
  content: string,
  options: RecallGenerationStateWriteOptions,
): Promise<void> {
  const filesystem = options.filesystem ?? nodeGenerationStateFilesystem;
  const directoryPath = dirname(destinationPath);
  await filesystem.createDirectory(directoryPath);
  const temporaryPath = `${destinationPath}.${randomUUID()}.tmp`;
  let file: RecallGenerationStateFile | undefined;
  try {
    file = await filesystem.openExclusiveFile(temporaryPath);
    await file.writeFile(content);
    await file.sync();
    await file.close();
    file = undefined;
    await filesystem.renameFile(temporaryPath, destinationPath);
    await filesystem.syncDirectory(directoryPath);
  } catch (error) {
    const cleanupErrors: unknown[] = [];
    if (file) {
      try {
        await file.close();
      } catch (closeError) {
        cleanupErrors.push(closeError);
      }
    }
    try {
      await filesystem.removeFile(temporaryPath);
    } catch (removeError) {
      cleanupErrors.push(removeError);
    }
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        'Recall atomic generation state write and cleanup failed',
      );
    }
    throw error;
  }
}

/** Atomically writes and durably publishes one checksummed active-generation pointer. */
export async function writeRecallActiveGenerationPointer(
  pointerPath: string,
  pointer: RecallActiveGenerationPointer,
  options: RecallGenerationStateWriteOptions = {},
): Promise<void> {
  await writeAtomicRecallGenerationState(
    pointerPath,
    encodeRecallActiveGenerationPointer(pointer),
    options,
  );
}

/** Reads one active pointer, returning null only when no pointer has ever been cut over. */
export async function readRecallActiveGenerationPointer(
  pointerPath: string,
): Promise<RecallActiveGenerationPointer | null> {
  try {
    return decodeRecallActiveGenerationPointer(await readFile(pointerPath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

/** Atomically writes and durably publishes one strict generation registry. */
export async function writeRecallGenerationRegistry(
  registryPath: string,
  registry: RecallGenerationRegistry,
  options: RecallGenerationStateWriteOptions = {},
): Promise<void> {
  await writeAtomicRecallGenerationState(
    registryPath,
    encodeRecallGenerationRegistry(registry),
    options,
  );
}

/** Atomically writes one scalar-only backlog summary after strict validation. */
export async function writeRecallBacklogSummary(
  backlogSummaryPath: string,
  summary: RecallBacklogSummary,
  options: RecallGenerationStateWriteOptions = {},
): Promise<void> {
  await writeAtomicRecallGenerationState(
    backlogSummaryPath,
    encodeRecallBacklogSummary(summary),
    options,
  );
}

/** Reads one generation registry, returning null only before generation management is initialized. */
export async function readRecallGenerationRegistry(
  registryPath: string,
): Promise<RecallGenerationRegistry | null> {
  try {
    return decodeRecallGenerationRegistry(await readFile(registryPath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

function isPathWithinRoot(candidatePath: string, rootPath: string): boolean {
  const relativePath = relative(rootPath, candidatePath);
  return (
    relativePath !== '' &&
    relativePath !== '..' &&
    !relativePath.startsWith(`..${sep}`) &&
    !isAbsolute(relativePath)
  );
}

/** Resolves one existing generation directory and rejects lexical or symlink root escape. */
export async function resolveRecallGenerationDirectory(
  generationRootDirectory: string,
  generationId: string,
): Promise<string> {
  if (!RECALL_GENERATION_IDENTIFIER_PATTERN.test(generationId)) {
    throw new Error('Recall generation identifier invalid');
  }
  const rootPath = resolve(generationRootDirectory);
  const candidatePath = resolve(rootPath, generationId);
  if (!isPathWithinRoot(candidatePath, rootPath)) {
    throw new Error('Recall generation directory escapes the configured generation root');
  }
  const [canonicalRootPath, canonicalCandidatePath] = await Promise.all([
    realpath(rootPath),
    realpath(candidatePath),
  ]);
  if (!isPathWithinRoot(canonicalCandidatePath, canonicalRootPath)) {
    throw new Error('Recall generation directory symlink escapes the configured generation root');
  }
  const generationStats = await stat(canonicalCandidatePath);
  if (!generationStats.isDirectory()) {
    throw new Error('selected generation is not a directory');
  }
  return canonicalCandidatePath;
}

/** Reads and validates the only pointer-selected generation directory available to search. */
export async function readRecallActiveGenerationSelection(
  activeGenerationPointerPath: string,
  generationRootDirectory: string,
): Promise<RecallActiveGenerationSelection> {
  try {
    const pointer = decodeRecallActiveGenerationPointer(
      await readFile(activeGenerationPointerPath, 'utf8'),
    );
    const generationDirectory = await resolveRecallGenerationDirectory(
      generationRootDirectory,
      pointer.activeGenerationId,
    );
    return {
      activeGenerationId: pointer.activeGenerationId,
      generationDirectory,
      databasePath: join(generationDirectory, 'zvec'),
      projectionDatabasePath: join(generationDirectory, 'session-projections'),
      statePath: join(generationDirectory, 'index-state.json'),
      manifestPath: join(generationDirectory, 'index-manifest.json'),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new RecallGenerationPointerError(message, { cause: error });
  }
}

/** Reads one scalar summary and returns a warning only for material recall backlog states. */
export async function readRecallMaterialBacklogWarning(
  backlogSummaryPath: string,
  activeGenerationId: string,
): Promise<string | null> {
  let summary: RecallBacklogSummary;
  try {
    summary = decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  if (summary.activeGenerationId !== activeGenerationId) {
    return null;
  }
  const materiallyStale =
    summary.oldestEligibleMarkerAgeMilliseconds !== null &&
    summary.oldestEligibleMarkerAgeMilliseconds >
      RECALL_MATERIAL_BACKLOG_SERVICE_OBJECTIVE_MILLISECONDS;
  const failed =
    summary.lastFailureCategory !== null ||
    summary.generationState === RecallGenerationCutoverState.FAILED;
  const rebuildingOnOlderGeneration =
    summary.buildingGenerationId !== null &&
    (summary.generationState === RecallGenerationCutoverState.BUILDING ||
      summary.generationState === RecallGenerationCutoverState.READY);
  const legacyReadOnly = summary.generationState === RecallGenerationCutoverState.LEGACY_READ_ONLY;
  const replayPending = summary.generationState === RecallGenerationCutoverState.REPLAY_PENDING;
  if (
    !materiallyStale &&
    !failed &&
    !rebuildingOnOlderGeneration &&
    !legacyReadOnly &&
    !replayPending
  ) {
    return null;
  }
  const scalarFields = [
    `pendingEligibleSessionCount=${summary.pendingEligibleSessionCount}`,
    `oldestEligibleAgeMilliseconds=${summary.oldestEligibleMarkerAgeMilliseconds ?? 'none'}`,
    `activeGenerationAgeMilliseconds=${summary.activeGenerationAgeMilliseconds}`,
    `generationState=${summary.generationState}`,
  ];
  if (rebuildingOnOlderGeneration || legacyReadOnly) {
    scalarFields.push(`rebuildAgeMilliseconds=${summary.rebuildAgeMilliseconds ?? 'none'}`);
  }
  if (summary.lastFailureCategory !== null) {
    scalarFields.push(`lastFailureCategory=${summary.lastFailureCategory}`);
  }
  return `Recall material backlog: ${scalarFields.join(' ')}`;
}
