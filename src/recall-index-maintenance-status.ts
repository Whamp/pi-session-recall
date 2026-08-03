import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const ISO_TIMESTAMP_PATTERN = '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}Z$';

const RECALL_INDEX_MAINTENANCE_STATUS_SCHEMA = Type.Object(
  {
    version: Type.Literal(1),
    completedAt: Type.String({ pattern: ISO_TIMESTAMP_PATTERN }),
    scannedSessions: Type.Integer({ minimum: 0 }),
    failedSessions: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

/** Last normally completed Index maintenance operation, recorded in UTC. */
export interface RecallIndexMaintenanceStatus {
  version: 1;
  completedAt: string;
  scannedSessions: number;
  failedSessions: number;
}

function isValidIndexMaintenanceTimestamp(value: string): boolean {
  const parsed = new Date(value);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString() === value;
}

/** Reads a compatible Index maintenance status, returning null for absent or invalid content. */
export async function readRecallIndexMaintenanceStatus(
  statusPath: string,
): Promise<RecallIndexMaintenanceStatus | null> {
  let content: string;
  try {
    content = await readFile(statusPath, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }

  try {
    const parsed: unknown = JSON.parse(content);
    const status = Value.Parse(RECALL_INDEX_MAINTENANCE_STATUS_SCHEMA, parsed);
    return isValidIndexMaintenanceTimestamp(status.completedAt) ? status : null;
  } catch {
    return null;
  }
}

/** Atomically publishes one normally completed Index maintenance status. */
export async function writeRecallIndexMaintenanceStatus(
  statusPath: string,
  status: RecallIndexMaintenanceStatus,
): Promise<void> {
  Value.Parse(RECALL_INDEX_MAINTENANCE_STATUS_SCHEMA, status);
  if (!isValidIndexMaintenanceTimestamp(status.completedAt)) {
    throw new Error(`Recall Index maintenance status timestamp invalid: ${status.completedAt}`);
  }
  await mkdir(dirname(statusPath), { recursive: true });
  const temporaryPath = `${statusPath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(status, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
    await rename(temporaryPath, statusPath);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}
