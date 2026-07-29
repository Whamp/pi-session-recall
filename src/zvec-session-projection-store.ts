import { existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecDoc,
  type ZVecStatus,
} from '@zvec/zvec';

import { RecallProjectionEncodingStatus, RecallSessionProjectionKind } from './enums.js';
import {
  decodeRecallSessionProjection,
  encodeRecallSessionProjection,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type PhysicalSessionProjection,
  type RecallSessionProjection,
} from './recall-session-projection.js';

/** Version of the scalar-only physical/logical projection collection schema. */
export const ZVEC_SESSION_PROJECTION_SCHEMA_VERSION = 1;

/** Checked scalar projection writes and strict generation-scoped reads. */
export interface ZvecSessionProjectionStore {
  upsertProjections(projections: readonly RecallSessionProjection[]): Promise<void>;
  deleteProjections(projectionIds: readonly string[]): Promise<void>;
  fetchProjections(projectionIds: readonly string[]): Map<string, RecallSessionProjection>;
  listPhysicalProjections(): PhysicalSessionProjection[];
  close(): void;
}

const PROJECTION_FIELD_SCHEMAS = [
  { name: 'schemaVersion', dataType: ZVecDataType.INT32 },
  { name: 'projectionKind', dataType: ZVecDataType.STRING },
  { name: 'generationId', dataType: ZVecDataType.STRING },
  { name: 'physicalSessionProjectionId', dataType: ZVecDataType.STRING },
  { name: 'logicalSessionId', dataType: ZVecDataType.STRING },
  { name: 'projectionJson', dataType: ZVecDataType.STRING },
];
const PROJECTION_OUTPUT_FIELDS = PROJECTION_FIELD_SCHEMAS.map(({ name }) => name);

function assertProjectionCollectionScalarSchema(collection: ZVecCollection): void {
  const storedFields = new Map(
    collection.schema.fields().map(({ name, dataType }) => [name, dataType]),
  );
  if (
    storedFields.size !== PROJECTION_FIELD_SCHEMAS.length ||
    PROJECTION_FIELD_SCHEMAS.some(({ name, dataType }) => storedFields.get(name) !== dataType)
  ) {
    throw new Error(
      `Recall projection zvec scalar schema mismatch: expected schema version ${ZVEC_SESSION_PROJECTION_SCHEMA_VERSION}`,
    );
  }
}

function readProjectionStringField(fields: Record<string, unknown>, name: string): string {
  const value = fields[name];
  if (typeof value !== 'string') {
    throw new Error(`Recall projection zvec field ${name} invalid: expected string`);
  }
  return value;
}

function readProjectionNumberField(fields: Record<string, unknown>, name: string): number {
  const value = fields[name];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Recall projection zvec field ${name} invalid: expected finite number`);
  }
  return value;
}

function serializeSessionProjection(projection: RecallSessionProjection): Record<string, unknown> {
  const encoded = encodeRecallSessionProjection(projection);
  if (encoded.status !== RecallProjectionEncodingStatus.ENCODED) {
    throw new Error(
      `Recall projection zvec write requires reconciliation: ${projection.projectionId} is ${encoded.byteLength} bytes`,
    );
  }
  return {
    schemaVersion: ZVEC_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: projection.projectionKind,
    generationId: projection.generationId,
    physicalSessionProjectionId:
      projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION
        ? projection.projectionId
        : projection.physicalProjectionId,
    logicalSessionId:
      projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
        ? projection.logicalSessionId
        : '',
    projectionJson: encoded.payload.projectionJson,
  };
}

function deserializeSessionProjection(doc: ZVecDoc, generationId: string): RecallSessionProjection {
  const fields: Record<string, unknown> = doc.fields;
  const schemaVersion = readProjectionNumberField(fields, 'schemaVersion');
  if (schemaVersion !== ZVEC_SESSION_PROJECTION_SCHEMA_VERSION) {
    throw new Error(
      `Recall projection zvec schema version mismatch: expected ${ZVEC_SESSION_PROJECTION_SCHEMA_VERSION}, received ${schemaVersion}`,
    );
  }
  const projectionKind = readProjectionStringField(fields, 'projectionKind');
  const projectionJson = readProjectionStringField(fields, 'projectionJson');
  const storedGenerationId = readProjectionStringField(fields, 'generationId');
  const projection = decodeRecallSessionProjection(
    {
      schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
      projectionKind,
      projectionId: doc.id,
      generationId: storedGenerationId,
      projectionJson,
    },
    { expectedGenerationId: generationId },
  );
  const expectedPhysicalProjectionId =
    projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION
      ? projection.projectionId
      : projection.physicalProjectionId;
  if (
    readProjectionStringField(fields, 'physicalSessionProjectionId') !==
    expectedPhysicalProjectionId
  ) {
    throw new Error(`Recall projection zvec physical identity mismatch: ${doc.id}`);
  }
  const expectedLogicalSessionId =
    projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
      ? projection.logicalSessionId
      : '';
  if (readProjectionStringField(fields, 'logicalSessionId') !== expectedLogicalSessionId) {
    throw new Error(`Recall projection zvec logical identity mismatch: ${doc.id}`);
  }
  return projection;
}

function assertCheckedZvecStatuses(
  operation: string,
  ids: readonly string[],
  statuses: readonly ZVecStatus[],
): void {
  if (statuses.length !== ids.length) {
    throw new Error(
      `Recall ${operation} status count mismatch: expected ${ids.length}, received ${statuses.length}`,
    );
  }
  for (const [index, status] of statuses.entries()) {
    if (!status.ok) {
      throw new Error(
        `Recall ${operation} failed at position ${index} for ${ids[index] ?? 'unknown'} [${status.code}]: ${status.message}`,
      );
    }
  }
}

/** Opens one generation's scalar-only projection collection without any vector field. */
export function openZvecSessionProjectionStore(config: {
  databasePath: string;
  generationId: string;
  createIfMissing?: boolean;
  readOnly?: boolean;
}): ZvecSessionProjectionStore {
  const databaseExists = existsSync(config.databasePath);
  if (!databaseExists && config.createIfMissing === false) {
    throw new Error(`Recall session projection collection missing at ${config.databasePath}`);
  }
  if (!databaseExists) {
    mkdirSync(dirname(config.databasePath), { recursive: true });
  }
  const collection: ZVecCollection = databaseExists
    ? config.readOnly
      ? ZVecOpen(config.databasePath, { readOnly: true })
      : ZVecOpen(config.databasePath)
    : ZVecCreateAndOpen(
        config.databasePath,
        new ZVecCollectionSchema({
          name: 'pi_session_recall_projections',
          fields: PROJECTION_FIELD_SCHEMAS,
        }),
      );
  if (collection.schema.vectors().length !== 0) {
    collection.closeSync();
    throw new Error('Recall session projection collection invalid: vector fields are forbidden');
  }
  try {
    assertProjectionCollectionScalarSchema(collection);
  } catch (error) {
    collection.closeSync();
    throw error instanceof Error
      ? error
      : new Error('Recall projection zvec scalar schema validation failed', { cause: error });
  }

  return {
    async upsertProjections(projections) {
      if (projections.length === 0) {
        return;
      }
      for (const projection of projections) {
        if (projection.generationId !== config.generationId) {
          throw new Error(
            `Recall projection zvec generation mismatch: expected ${config.generationId}, received ${projection.generationId}`,
          );
        }
      }
      const ids = projections.map(({ projectionId }) => projectionId);
      const statuses = collection.upsertSync(
        projections.map((projection) => ({
          id: projection.projectionId,
          fields: serializeSessionProjection(projection),
        })),
      );
      assertCheckedZvecStatuses('projection upsert', ids, statuses);
    },
    async deleteProjections(projectionIds) {
      if (projectionIds.length === 0) {
        return;
      }
      const ids = [...projectionIds];
      assertCheckedZvecStatuses('projection delete', ids, collection.deleteSync(ids));
    },
    fetchProjections(projectionIds) {
      if (projectionIds.length === 0) {
        return new Map();
      }
      const docs = collection.fetchSync({
        ids: [...projectionIds],
        outputFields: PROJECTION_OUTPUT_FIELDS,
        includeVector: false,
      });
      return new Map(
        Object.values(docs).map((doc) => [
          doc.id,
          deserializeSessionProjection(doc, config.generationId),
        ]),
      );
    },
    listPhysicalProjections() {
      if (collection.stats.docCount === 0) {
        return [];
      }
      return collection
        .querySync({
          filter: `projectionKind = '${RecallSessionProjectionKind.PHYSICAL_SESSION}'`,
          topk: collection.stats.docCount,
          outputFields: PROJECTION_OUTPUT_FIELDS,
          includeVector: false,
        })
        .map((doc) => deserializeSessionProjection(doc, config.generationId))
        .filter(
          (projection): projection is PhysicalSessionProjection =>
            projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
        )
        .toSorted((left, right) => left.projectionId.localeCompare(right.projectionId));
    },
    close() {
      collection.closeSync();
    },
  };
}
