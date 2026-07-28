import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallBacklogFailureCategory, RecallGenerationCutoverState } from './enums.js';
import { RecallGenerationPointerError } from './errors.js';
import { RECALL_INDEX_MANIFEST_VERSION } from './recall-index-manifest.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import { RECALL_WORK_MARKER_VERSION } from './recall-work-marker.js';

/** Current strict version for the checksummed active-generation pointer. */
export const RECALL_ACTIVE_GENERATION_POINTER_VERSION = 1;

/** Current strict version for the generation build and cutover registry. */
export const RECALL_GENERATION_REGISTRY_VERSION = 1;

/** Current strict version for the scalar material recall backlog summary. */
export const RECALL_BACKLOG_SUMMARY_VERSION = 1;

/** Service objective starts after the longest expected 30-minute quiescence window. */
export const RECALL_MATERIAL_BACKLOG_SERVICE_OBJECTIVE_MILLISECONDS = 30 * 60_000;

/** Pointer-selected immutable paths used by one read-only recall search. */
export interface RecallActiveGenerationSelection {
  activeGenerationId: string;
  generationDirectory: string;
  databasePath: string;
  projectionDatabasePath: string;
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

/** One generation's durable explicit-rebuild and cutover state. */
export interface RecallGenerationRegistryEntry {
  generationId: string;
  state: RecallGenerationCutoverState;
  indexManifestVersion: 6;
  markerSchemaVersion: 1;
  sessionProjectionSchemaVersion: 2;
  indexManifestFingerprint: string;
  rebuildStartedAtEpochMilliseconds: number;
  stateChangedAtEpochMilliseconds: number;
  rebuildStartMarkerId: string | null;
}

/** Durable registry of active, optional building, and bounded rollback generations. */
export interface RecallGenerationRegistry {
  version: 1;
  activeGenerationId: string;
  buildingGenerationId: string | null;
  rollbackGenerationId: string | null;
  activePointerChecksum: string;
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

const nonemptyStringSchema = Type.String({ minLength: 1 });
const generationIdentifierSchema = Type.String({ pattern: '^[A-Za-z0-9_-]+$' });
const nullableIdentifierSchema = Type.Union([nonemptyStringSchema, Type.Null()]);
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
    generationId: nonemptyStringSchema,
    state: Type.Enum(RecallGenerationCutoverState),
    indexManifestVersion: Type.Literal(RECALL_INDEX_MANIFEST_VERSION),
    markerSchemaVersion: Type.Literal(RECALL_WORK_MARKER_VERSION),
    sessionProjectionSchemaVersion: Type.Literal(RECALL_SESSION_PROJECTION_SCHEMA_VERSION),
    indexManifestFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    rebuildStartedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    stateChangedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    rebuildStartMarkerId: nullableIdentifierSchema,
  },
  { additionalProperties: false },
);
const recallGenerationRegistrySchema = Type.Object(
  {
    version: Type.Literal(RECALL_GENERATION_REGISTRY_VERSION),
    activeGenerationId: nonemptyStringSchema,
    buildingGenerationId: nullableIdentifierSchema,
    rollbackGenerationId: nullableIdentifierSchema,
    activePointerChecksum: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    generations: Type.Array(recallGenerationRegistryEntrySchema, { minItems: 1 }),
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
      JSON.stringify({
        version: pointer.version,
        activeGenerationId: pointer.activeGenerationId,
      }),
    )
    .digest('hex');
}

/** Creates one complete checksummed active-generation pointer without writing it. */
export function createRecallActiveGenerationPointer(
  activeGenerationId: string,
): RecallActiveGenerationPointer {
  if (activeGenerationId.length === 0) {
    throw new Error('Recall active generation pointer generation ID must not be empty');
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

function assertRecallGenerationRegistryInvariants(registry: RecallGenerationRegistry): void {
  const generationIds = registry.generations.map(({ generationId }) => generationId);
  if (new Set(generationIds).size !== generationIds.length) {
    throw new Error('Recall generation registry contains a duplicate generation ID');
  }
  const expectedPointer = createRecallActiveGenerationPointer(registry.activeGenerationId);
  if (registry.activePointerChecksum !== expectedPointer.checksum) {
    throw new Error(
      `Recall generation registry active pointer checksum mismatch: expected ${expectedPointer.checksum}, received ${registry.activePointerChecksum}`,
    );
  }
  const active = findRegistryGeneration(registry, registry.activeGenerationId, 'active');
  if (active.state !== RecallGenerationCutoverState.ACTIVE) {
    throw new Error(
      `Recall generation registry active generation has invalid state: ${active.state}`,
    );
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

/** Reads and validates the only pointer-selected generation directory available to search. */
export async function readRecallActiveGenerationSelection(
  activeGenerationPointerPath: string,
  generationRootDirectory: string,
): Promise<RecallActiveGenerationSelection> {
  try {
    const pointer = decodeRecallActiveGenerationPointer(
      await readFile(activeGenerationPointerPath, 'utf8'),
    );
    const generationDirectory = join(generationRootDirectory, pointer.activeGenerationId);
    const generationStats = await stat(generationDirectory);
    if (!generationStats.isDirectory()) {
      throw new Error('selected generation is not a directory');
    }
    return {
      activeGenerationId: pointer.activeGenerationId,
      generationDirectory,
      databasePath: join(generationDirectory, 'zvec'),
      projectionDatabasePath: join(generationDirectory, 'session-projections'),
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
    const errorCode =
      typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined;
    if (errorCode === 'ENOENT') {
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
  if (!materiallyStale && !failed && !rebuildingOnOlderGeneration) {
    return null;
  }
  const scalarFields = [
    `pendingEligibleSessionCount=${summary.pendingEligibleSessionCount}`,
    `oldestEligibleAgeMilliseconds=${summary.oldestEligibleMarkerAgeMilliseconds ?? 'none'}`,
    `activeGenerationAgeMilliseconds=${summary.activeGenerationAgeMilliseconds}`,
    `generationState=${summary.generationState}`,
  ];
  if (rebuildingOnOlderGeneration) {
    scalarFields.push(`rebuildAgeMilliseconds=${summary.rebuildAgeMilliseconds ?? 'none'}`);
  }
  if (summary.lastFailureCategory !== null) {
    scalarFields.push(`lastFailureCategory=${summary.lastFailureCategory}`);
  }
  return `Recall material backlog: ${scalarFields.join(' ')}`;
}
