import { createHash } from 'node:crypto';
import { open, stat } from 'node:fs/promises';

import { RecallAppendDeltaStatus, RecallProjectionRepairReason } from './enums.js';
import {
  isSupportedRecallSessionHeader,
  parseRecallSessionRecord,
  type ParsedRecallSessionRecord,
} from './parse-recall-session-record.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';

const RECALL_APPEND_BOUNDARY_WINDOW_BYTES = 4_096;

/** Injectable bounded range reader used to verify which source bytes append projection consumes. */
export interface RecallSessionAppendReadOptions {
  readRange?: (
    sourcePath: string,
    startByte: number,
    endByteExclusive: number,
  ) => AsyncIterable<Buffer>;
}

/** Successfully framed complete append records and the next durable physical cursor. */
export interface RecallSessionAppendDelta {
  status: RecallAppendDeltaStatus.APPENDED;
  records: ParsedRecallSessionRecord[];
  appendCursorBytes: number;
  appendCursorLines: number;
  boundaryFingerprint: string;
  partialFinalRecordBytes: number;
  sourceDevice: string;
  sourceInode: string;
}

/** Actionable append failure that delegates repair to explicit full reconciliation. */
export interface RecallSessionAppendReconciliation {
  status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION;
  repairReason: RecallProjectionRepairReason;
}

/** Strict append result; malformed or rewritten sources never produce speculative records. */
export type RecallSessionAppendDeltaResult =
  | RecallSessionAppendDelta
  | RecallSessionAppendReconciliation;

async function* readRecallFileRange(
  sourcePath: string,
  startByte: number,
  endByteExclusive: number,
): AsyncGenerator<Buffer> {
  const handle = await open(sourcePath, 'r');
  try {
    let position = startByte;
    while (position < endByteExclusive) {
      const byteLength = Math.min(64 * 1_024, endByteExclusive - position);
      const bytes = Buffer.allocUnsafe(byteLength);
      const read = await handle.read(bytes, 0, byteLength, position);
      if (read.bytesRead === 0) {
        break;
      }
      position += read.bytesRead;
      yield bytes.subarray(0, read.bytesRead);
    }
  } finally {
    await handle.close();
  }
}

async function readRecallRangeBytes(
  readRange: NonNullable<RecallSessionAppendReadOptions['readRange']>,
  sourcePath: string,
  startByte: number,
  endByteExclusive: number,
): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let byteLength = 0;
  for await (const chunk of readRange(sourcePath, startByte, endByteExclusive)) {
    chunks.push(chunk);
    byteLength += chunk.length;
  }
  return Buffer.concat(chunks, byteLength);
}

function createRecallBoundaryFingerprint(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function reconciliation(
  repairReason: RecallProjectionRepairReason,
): RecallSessionAppendReconciliation {
  return { status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION, repairReason };
}

function frameCompleteRecallAppendRecords(
  bytes: Buffer,
  sourcePath: string,
  cursorBytes: number,
  cursorLines: number,
): Pick<
  RecallSessionAppendDelta,
  'records' | 'appendCursorBytes' | 'appendCursorLines' | 'partialFinalRecordBytes'
> {
  const records: ParsedRecallSessionRecord[] = [];
  let recordStart = 0;
  let completedLines = 0;
  let lineFeedIndex = bytes.indexOf(0x0a);
  while (lineFeedIndex !== -1) {
    completedLines += 1;
    const framed = bytes.subarray(recordStart, lineFeedIndex);
    const contentEnd = framed.at(-1) === 0x0d ? framed.length - 1 : framed.length;
    const content = framed.subarray(0, contentEnd);
    if (content.toString('utf8').trim()) {
      records.push(
        parseRecallSessionRecord(
          content.toString('utf8'),
          sourcePath,
          cursorLines + completedLines,
          cursorBytes + recordStart,
          cursorBytes + lineFeedIndex + 1,
        ),
      );
    }
    recordStart = lineFeedIndex + 1;
    lineFeedIndex = bytes.indexOf(0x0a, recordStart);
  }
  return {
    records,
    appendCursorBytes: cursorBytes + recordStart,
    appendCursorLines: cursorLines + completedLines,
    partialFinalRecordBytes: bytes.length - recordStart,
  };
}

function hasUnsupportedAppendLayout(
  projection: PhysicalSessionProjection,
  records: readonly ParsedRecallSessionRecord[],
): boolean {
  for (const record of records) {
    if (record.value.type === 'session' && !isSupportedRecallSessionHeader(record.value)) {
      return true;
    }
  }
  if (projection.appendCursorBytes === 0) {
    return records[0]?.value.type !== 'session';
  }
  return false;
}

/** Validates one durable append cursor and reads only its bounded fingerprint window plus appended bytes. */
export async function readRecallSessionAppendDelta(
  filePath: string,
  physicalProjection: PhysicalSessionProjection,
  options: RecallSessionAppendReadOptions = {},
): Promise<RecallSessionAppendDeltaResult> {
  const cursorBytes = physicalProjection.appendCursorBytes;
  const cursorLines = physicalProjection.appendCursorLines;
  if (!Number.isSafeInteger(cursorBytes) || cursorBytes < 0 || !Number.isSafeInteger(cursorLines)) {
    return reconciliation(RecallProjectionRepairReason.APPEND_CURSOR_MISSING);
  }
  if (filePath !== physicalProjection.sourcePath) {
    return reconciliation(RecallProjectionRepairReason.SOURCE_IDENTITY_MISMATCH);
  }
  const metadata = await stat(filePath, { bigint: true });
  const sourceDevice = metadata.dev.toString();
  const sourceInode = metadata.ino.toString();
  if (
    sourceDevice !== physicalProjection.sourceDevice ||
    sourceInode !== physicalProjection.sourceInode
  ) {
    return reconciliation(RecallProjectionRepairReason.SOURCE_IDENTITY_MISMATCH);
  }
  const sourceSize = Number(metadata.size);
  if (!Number.isSafeInteger(sourceSize) || sourceSize < cursorBytes) {
    return reconciliation(RecallProjectionRepairReason.SOURCE_SHRANK);
  }

  const readRange = options.readRange ?? readRecallFileRange;
  const boundaryStart = Math.max(0, cursorBytes - RECALL_APPEND_BOUNDARY_WINDOW_BYTES);
  const boundaryBytes =
    boundaryStart === cursorBytes
      ? Buffer.alloc(0)
      : await readRecallRangeBytes(readRange, filePath, boundaryStart, cursorBytes);
  if (
    boundaryBytes.length !== cursorBytes - boundaryStart ||
    createRecallBoundaryFingerprint(boundaryBytes) !== physicalProjection.boundaryFingerprint
  ) {
    return reconciliation(RecallProjectionRepairReason.BOUNDARY_MISMATCH);
  }

  const appendBytes =
    sourceSize === cursorBytes
      ? Buffer.alloc(0)
      : await readRecallRangeBytes(readRange, filePath, cursorBytes, sourceSize);
  if (appendBytes.length !== sourceSize - cursorBytes) {
    return reconciliation(RecallProjectionRepairReason.APPEND_CURSOR_MISSING);
  }
  let framed: ReturnType<typeof frameCompleteRecallAppendRecords>;
  try {
    framed = frameCompleteRecallAppendRecords(appendBytes, filePath, cursorBytes, cursorLines);
  } catch {
    return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
  }
  if (hasUnsupportedAppendLayout(physicalProjection, framed.records)) {
    return reconciliation(RecallProjectionRepairReason.UNSUPPORTED_LAYOUT);
  }
  const committedBoundaryStart = Math.max(
    0,
    framed.appendCursorBytes - RECALL_APPEND_BOUNDARY_WINDOW_BYTES,
  );
  let committedBoundaryBytes: Buffer;
  if (committedBoundaryStart >= cursorBytes) {
    committedBoundaryBytes = appendBytes.subarray(
      committedBoundaryStart - cursorBytes,
      framed.appendCursorBytes - cursorBytes,
    );
  } else {
    const prefix = boundaryBytes.subarray(committedBoundaryStart - boundaryStart);
    const suffix = appendBytes.subarray(0, framed.appendCursorBytes - cursorBytes);
    committedBoundaryBytes = Buffer.concat([prefix, suffix]);
  }
  return {
    status: RecallAppendDeltaStatus.APPENDED,
    ...framed,
    boundaryFingerprint: createRecallBoundaryFingerprint(committedBoundaryBytes),
    sourceDevice,
    sourceInode,
  };
}
