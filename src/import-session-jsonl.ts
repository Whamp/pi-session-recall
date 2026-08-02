import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';

import { SessionImportFormat } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';

/** Versioned identity of exact session framing, detection, and virtual conversion policy. */
export const SESSION_IMPORT_POLICY_VERSION = 3;

/** One parsed physical JSONL record with its trustworthy one-based source line. */
export interface PhysicalSessionJsonlRecord {
  sourceLine: number;
  value: Record<string, unknown>;
}

/** One canonical logical session produced without mutating its physical source file. */
export interface CanonicalSessionRepresentation {
  format: SessionImportFormat;
  physicalPath: string;
  logicalSessionId: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  records: PhysicalSessionJsonlRecord[];
}

/** Detected physical format and canonical logical sessions ready for strict graph validation. */
export interface SessionJsonlImport {
  format: SessionImportFormat;
  sessions: CanonicalSessionRepresentation[];
}

function decodeSessionJsonlRecord(
  recordParts: Buffer[],
  recordByteLength: number,
  endedByLineFeed: boolean,
): string {
  const record = Buffer.concat(recordParts, recordByteLength);
  const contentEnd = endedByLineFeed && record.at(-1) === 0x0d ? record.length - 1 : record.length;
  return record.subarray(0, contentEnd).toString('utf8');
}

async function* frameSessionJsonlRecords(
  sessionPath: string,
): AsyncGenerator<{ sourceLine: number; text: string }> {
  const recordParts: Buffer[] = [];
  let recordByteLength = 0;
  let sourceLine = 0;
  for await (const streamChunk of createReadStream(sessionPath)) {
    const chunk = Buffer.isBuffer(streamChunk) ? streamChunk : Buffer.from(String(streamChunk));
    let recordStart = 0;
    let lineFeedIndex = chunk.indexOf(0x0a, recordStart);
    while (lineFeedIndex !== -1) {
      const recordPart = chunk.subarray(recordStart, lineFeedIndex);
      recordParts.push(recordPart);
      recordByteLength += recordPart.length;
      sourceLine += 1;
      yield {
        sourceLine,
        text: decodeSessionJsonlRecord(recordParts, recordByteLength, true),
      };
      recordParts.length = 0;
      recordByteLength = 0;
      recordStart = lineFeedIndex + 1;
      lineFeedIndex = chunk.indexOf(0x0a, recordStart);
    }
    const trailingPart = chunk.subarray(recordStart);
    if (trailingPart.length > 0) {
      recordParts.push(trailingPart);
      recordByteLength += trailingPart.length;
    }
  }
  if (recordByteLength > 0) {
    sourceLine += 1;
    yield {
      sourceLine,
      text: decodeSessionJsonlRecord(recordParts, recordByteLength, false),
    };
  }
}

async function readPhysicalSessionJsonlRecords(
  sessionPath: string,
): Promise<PhysicalSessionJsonlRecord[]> {
  const records: PhysicalSessionJsonlRecord[] = [];
  for await (const framed of frameSessionJsonlRecords(sessionPath)) {
    if (!framed.text.trim()) {
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(framed.text);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall session JSON invalid at ${sessionPath}:${framed.sourceLine}: ${message}`,
        { cause: error },
      );
    }
    if (!isUnknownRecord(parsed) || typeof parsed.type !== 'string') {
      throw new Error(
        `Recall session import unsupported or ambiguous at ${sessionPath}:${framed.sourceLine}: each record must be an object with a type`,
      );
    }
    records.push({ sourceLine: framed.sourceLine, value: parsed });
  }
  return records;
}

function isNonemptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function isCompleteSessionHeader(record: Record<string, unknown>): boolean {
  return (
    record.type === 'session' &&
    isNonemptyString(record.id) &&
    isNonemptyString(record.timestamp) &&
    typeof record.cwd === 'string'
  );
}

function isValidV1Message(message: unknown): boolean {
  if (
    !isUnknownRecord(message) ||
    !isNonemptyString(message.role) ||
    typeof message.timestamp !== 'number'
  ) {
    return false;
  }
  if (message.role === 'user') {
    return Object.hasOwn(message, 'content');
  }
  if (message.role === 'assistant') {
    return (
      Array.isArray(message.content) &&
      isNonemptyString(message.api) &&
      isNonemptyString(message.provider) &&
      isNonemptyString(message.model) &&
      isUnknownRecord(message.usage) &&
      isNonemptyString(message.stopReason)
    );
  }
  if (message.role === 'toolResult') {
    return (
      Array.isArray(message.content) &&
      isNonemptyString(message.toolCallId) &&
      isNonemptyString(message.toolName) &&
      typeof message.isError === 'boolean'
    );
  }
  if (message.role === 'bashExecution') {
    return (
      typeof message.command === 'string' &&
      typeof message.output === 'string' &&
      typeof message.cancelled === 'boolean' &&
      typeof message.truncated === 'boolean'
    );
  }
  return false;
}

function isValidV1LinearEntry(record: Record<string, unknown>): boolean {
  if (
    Object.hasOwn(record, 'id') ||
    Object.hasOwn(record, 'parentId') ||
    !isNonemptyString(record.timestamp)
  ) {
    return false;
  }
  if (record.type === 'message') {
    return isValidV1Message(record.message);
  }
  if (record.type === 'model_change') {
    return isNonemptyString(record.provider) && isNonemptyString(record.modelId);
  }
  if (record.type === 'thinking_level_change') {
    return isNonemptyString(record.thinkingLevel);
  }
  if (record.type === 'compaction') {
    return (
      typeof record.summary === 'string' &&
      Number.isInteger(record.firstKeptEntryIndex) &&
      typeof record.tokensBefore === 'number'
    );
  }
  return false;
}

function isPiV1LinearSession(records: PhysicalSessionJsonlRecord[]): boolean {
  const header = records[0]?.value;
  const entries = records.slice(1);
  return Boolean(
    header &&
    isCompleteSessionHeader(header) &&
    !Object.hasOwn(header, 'version') &&
    entries.length > 0 &&
    entries.every(({ value }) => isValidV1LinearEntry(value)),
  );
}

function createDeterministicV1EntryId(
  sessionId: string,
  record: PhysicalSessionJsonlRecord,
): string {
  return createHash('sha256')
    .update(`pi-v1-entry-v1\0${sessionId}\0${record.sourceLine}\0${JSON.stringify(record.value)}`)
    .digest('hex')
    .slice(0, 40);
}

function omitV1CompactionIndex(record: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(record).filter(([fieldName]) => fieldName !== 'firstKeptEntryIndex'),
  );
}

function convertPiV1LinearSession(
  sessionPath: string,
  records: PhysicalSessionJsonlRecord[],
): CanonicalSessionRepresentation {
  const headerRecord = records[0];
  if (!headerRecord || !isNonemptyString(headerRecord.value.id)) {
    throw new Error(`Recall v1 session header missing at ${sessionPath}`);
  }
  const sessionId = headerRecord.value.id;
  const sourceEntries = records.slice(1);
  const entryIds = sourceEntries.map((record) => createDeterministicV1EntryId(sessionId, record));
  if (new Set(entryIds).size !== entryIds.length) {
    throw new Error(`Recall v1 deterministic entry ID collision at ${sessionPath}`);
  }
  const convertedEntries = sourceEntries.map((record, entryIndex) => {
    const id = entryIds[entryIndex];
    if (!id) {
      throw new Error(
        `Recall v1 deterministic entry ID missing at ${sessionPath}:${record.sourceLine}`,
      );
    }
    const previousId = entryIds[entryIndex - 1] ?? null;
    let convertedValue: Record<string, unknown> = record.value;
    if (record.value.type === 'compaction') {
      const firstKeptEntryIndex = record.value.firstKeptEntryIndex;
      const firstKeptEntryId =
        typeof firstKeptEntryIndex === 'number' && Number.isInteger(firstKeptEntryIndex)
          ? entryIds[firstKeptEntryIndex - 1]
          : undefined;
      if (!firstKeptEntryId) {
        throw new Error(
          `Recall v1 session invalid at ${sessionPath}:${record.sourceLine}: compaction firstKeptEntryIndex does not name an entry`,
        );
      }
      convertedValue = {
        ...omitV1CompactionIndex(record.value),
        firstKeptEntryId,
      };
    }
    return {
      sourceLine: record.sourceLine,
      value: {
        ...convertedValue,
        id,
        parentId: previousId,
      },
    };
  });
  const lastRecord = records.at(-1);
  if (!lastRecord) {
    throw new Error(`Recall v1 session records missing at ${sessionPath}`);
  }
  return {
    format: SessionImportFormat.PI_V1_LINEAR,
    physicalPath: sessionPath,
    logicalSessionId: sessionId,
    sourceLineStart: headerRecord.sourceLine,
    sourceLineEnd: lastRecord.sourceLine,
    records: [
      {
        sourceLine: headerRecord.sourceLine,
        value: { ...headerRecord.value, version: 2 },
      },
      ...convertedEntries,
    ],
  };
}

function isSupportedCanonicalSessionVersion(version: unknown): boolean {
  return version === 2 || version === 3;
}

function isCanonicalReuseHeader(record: Record<string, unknown>): boolean {
  return isCompleteSessionHeader(record) && isSupportedCanonicalSessionVersion(record.version);
}

function splitPiSessionReuseHistory(
  sessionPath: string,
  records: PhysicalSessionJsonlRecord[],
): CanonicalSessionRepresentation[] | null {
  const headerIndexes = records.flatMap(({ value }, index) =>
    value.type === 'session' ? [index] : [],
  );
  if (headerIndexes.length < 2) {
    return null;
  }
  if (
    headerIndexes[0] !== 0 ||
    !headerIndexes.every((index) => {
      const header = records[index];
      return Boolean(header && isCanonicalReuseHeader(header.value));
    })
  ) {
    throw new Error(
      `Recall session import unsupported or ambiguous at ${sessionPath}: reuse history requires complete versioned headers and no pre-header records`,
    );
  }
  return headerIndexes.map((headerIndex, segmentIndex) => {
    const segmentEnd = headerIndexes[segmentIndex + 1] ?? records.length;
    const segment = records.slice(headerIndex, segmentEnd);
    const header = segment[0];
    const lastRecord = segment.at(-1);
    if (!header || !lastRecord || segment.length < 2 || !isNonemptyString(header.value.id)) {
      throw new Error(
        `Recall session import unsupported or ambiguous at ${sessionPath}: every reuse-history header must begin a nonempty logical session`,
      );
    }
    return {
      format: SessionImportFormat.PI_SESSION_REUSE_HISTORY,
      physicalPath: sessionPath,
      logicalSessionId: header.value.id,
      sourceLineStart: header.sourceLine,
      sourceLineEnd: lastRecord.sourceLine,
      records: segment,
    };
  });
}

function createCanonicalSingleSession(
  sessionPath: string,
  records: PhysicalSessionJsonlRecord[],
): CanonicalSessionRepresentation {
  const header = records[0];
  const lastRecord = records.at(-1);
  if (!header || !lastRecord || !isCompleteSessionHeader(header.value)) {
    throw new Error(
      `Recall session import unsupported or ambiguous at ${sessionPath}: expected one complete leading session header`,
    );
  }
  if (!isSupportedCanonicalSessionVersion(header.value.version)) {
    throw new Error(
      `Recall session import unsupported or ambiguous at ${sessionPath}:${header.sourceLine}: canonical session version must be 2 or 3`,
    );
  }
  return {
    format: SessionImportFormat.CANONICAL_JSONL,
    physicalPath: sessionPath,
    logicalSessionId: String(header.value.id),
    sourceLineStart: header.sourceLine,
    sourceLineEnd: lastRecord.sourceLine,
    records,
  };
}

/** Streams and frames one physical session file, then selects one exact virtual import path. */
export async function importSessionJsonl(sessionPath: string): Promise<SessionJsonlImport> {
  const records = await readPhysicalSessionJsonlRecords(sessionPath);
  if (records.length === 0) {
    throw new Error(`Recall session import unsupported or ambiguous at ${sessionPath}: no records`);
  }
  const reuseSessions = splitPiSessionReuseHistory(sessionPath, records);
  if (reuseSessions) {
    return {
      format: SessionImportFormat.PI_SESSION_REUSE_HISTORY,
      sessions: reuseSessions,
    };
  }
  if (isPiV1LinearSession(records)) {
    return {
      format: SessionImportFormat.PI_V1_LINEAR,
      sessions: [convertPiV1LinearSession(sessionPath, records)],
    };
  }
  return {
    format: SessionImportFormat.CANONICAL_JSONL,
    sessions: [createCanonicalSingleSession(sessionPath, records)],
  };
}
