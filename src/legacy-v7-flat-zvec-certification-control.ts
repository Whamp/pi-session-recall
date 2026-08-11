import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';

import {
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
} from '@zvec/zvec';

import type { ProjectIdentity } from './resolve-project-identity.js';

/** Read-only query surface retained solely to compare v8 against the superseded v7 control. */
export interface LegacyV7FlatZvecCertificationControl {
  searchDocumentIds(
    embedding: number[],
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): string[];
  close(): void;
}

function assertLegacyV7FlatControlSchema(
  collection: ZVecCollection,
  databasePath: string,
  dimensions: number,
): void {
  const embedding = collection.schema.vector('embedding');
  const projectIdentityDigest = collection.schema
    .fields()
    .find((field) => field.name === 'projectIdentityDigest');
  if (
    collection.schema.vectors().length !== 1 ||
    embedding.dataType !== ZVecDataType.VECTOR_FP32 ||
    embedding.dimension !== dimensions ||
    embedding.indexParams?.indexType !== ZVecIndexType.FLAT ||
    embedding.indexParams.metricType !== ZVecMetricType.IP ||
    projectIdentityDigest?.dataType !== ZVecDataType.STRING
  ) {
    throw new Error(
      `Legacy-v7 flat-Zvec certification control schema incompatible at ${databasePath}`,
    );
  }
}

function createProjectIdentityDigestFilter(projectIdentity?: ProjectIdentity): string | undefined {
  if (!projectIdentity) {
    return undefined;
  }
  const digest = createHash('sha256').update(projectIdentity).digest('hex');
  return `projectIdentityDigest = '${digest}'`;
}

/** Opens the superseded v7 dense-only FLAT store without exposing it to production Recall paths. */
export function openLegacyV7FlatZvecCertificationControl(config: {
  databasePath: string;
  dimensions: number;
}): LegacyV7FlatZvecCertificationControl {
  if (!existsSync(config.databasePath)) {
    throw new Error(`Legacy-v7 flat-Zvec certification control missing at ${config.databasePath}`);
  }
  const collection = ZVecOpen(config.databasePath, { readOnly: true });
  try {
    assertLegacyV7FlatControlSchema(collection, config.databasePath, config.dimensions);
  } catch (error) {
    collection.closeSync();
    throw error;
  }

  return {
    searchDocumentIds(embedding, limit, projectIdentity) {
      if (embedding.length !== config.dimensions) {
        throw new Error(
          `Legacy-v7 flat-Zvec certification query dimension invalid: expected ${config.dimensions}, received ${embedding.length}`,
        );
      }
      if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
        throw new Error(
          'Legacy-v7 flat-Zvec certification candidate limit invalid: expected 1 through 200',
        );
      }
      const projectFilter = createProjectIdentityDigestFilter(projectIdentity);
      return collection
        .querySync({
          fieldName: 'embedding',
          vector: embedding,
          topk: limit,
          outputFields: [],
          includeVector: false,
          ...(projectFilter ? { filter: projectFilter } : {}),
        })
        .map(({ id }) => id);
    },
    close() {
      collection.closeSync();
    },
  };
}
