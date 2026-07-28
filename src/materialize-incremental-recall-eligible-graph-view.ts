import { stat } from 'node:fs/promises';

import { SessionImportFormat } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  isSupportedRecallSessionHeader,
  parseRecallSessionGraph,
  parseRecallSessionRecord,
  type CanonicalSessionRepresentation,
  type ParsedRecallSessionGraph,
  type ParsedRecallSessionRecord,
} from './parse-recall-session-record.js';
import type {
  LogicalSessionProjection,
  PhysicalSessionProjection,
  RecallEligibleSourceSpan,
  RecallProjectedEntryDescriptor,
} from './recall-session-projection.js';
import {
  readRecallSessionSourceRange,
  type RecallSessionAppendDelta,
  type RecallSessionSourceRangeReader,
} from './read-recall-session-append-delta.js';

/** A strictly validated graph view whose payload records are limited to newly eligible closure. */
export interface IncrementalRecallEligibleGraphView {
  graph: ParsedRecallSessionGraph;
  physicalPath: string;
  logicalSessionId: string;
}

/** Scalar projection, append records, and ranged-I/O inputs for one eligible graph view. */
export interface MaterializeIncrementalRecallEligibleGraphViewOptions {
  physicalProjection: PhysicalSessionProjection;
  logicalProjection: LogicalSessionProjection;
  newlyEligibleSpans: readonly RecallEligibleSourceSpan[];
  appendDelta: RecallSessionAppendDelta;
  readRange?: RecallSessionSourceRangeReader;
}

async function readExactRecallSourceRange(
  readRange: RecallSessionSourceRangeReader,
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
  const bytes = Buffer.concat(chunks, byteLength);
  if (bytes.length !== endByteExclusive - startByte) {
    throw new Error(
      `Recall incremental graph source range incomplete: ${startByte}-${endByteExclusive}`,
    );
  }
  return bytes;
}

function decodeProjectedRecallRecord(
  bytes: Buffer,
  sourcePath: string,
  descriptor: Pick<
    RecallProjectedEntryDescriptor,
    'sourceLine' | 'startByte' | 'endByte' | 'sourceFingerprint'
  >,
): ParsedRecallSessionRecord {
  const lineEnd = bytes.at(-1) === 0x0a ? bytes.length - 1 : bytes.length;
  const contentEnd = lineEnd > 0 && bytes.at(lineEnd - 1) === 0x0d ? lineEnd - 1 : lineEnd;
  const record = parseRecallSessionRecord(
    bytes.subarray(0, contentEnd).toString('utf8'),
    sourcePath,
    descriptor.sourceLine,
    descriptor.startByte,
    descriptor.endByte,
  );
  if (record.sourceFingerprint !== descriptor.sourceFingerprint) {
    throw new Error(
      `Recall incremental graph source fingerprint mismatch at ${sourcePath}:${descriptor.sourceLine}`,
    );
  }
  return record;
}

function validateMaterializedEntryRecord(
  record: ParsedRecallSessionRecord,
  descriptor: RecallProjectedEntryDescriptor,
): void {
  const value = record.value;
  const messageRole =
    value.type === 'message' &&
    isUnknownRecord(value.message) &&
    typeof value.message.role === 'string'
      ? value.message.role
      : null;
  if (
    value.id !== descriptor.entryId ||
    value.parentId !== descriptor.parentEntryId ||
    value.type !== descriptor.entryType ||
    value.timestamp !== descriptor.timestamp ||
    messageRole !== descriptor.messageRole
  ) {
    throw new Error(
      `Recall incremental graph entry scalar mismatch at ${record.sourceLine}: ${descriptor.entryId}`,
    );
  }
}

function createProjectedEntryPlaceholder(
  descriptor: RecallProjectedEntryDescriptor,
): Record<string, unknown> {
  const record: Record<string, unknown> = {
    type: descriptor.entryType,
    id: descriptor.entryId,
    parentId: descriptor.parentEntryId,
    timestamp: descriptor.timestamp,
  };
  if (descriptor.entryType === 'message') {
    let content: unknown[] = [];
    if (descriptor.toolCalls.length > 0) {
      content = descriptor.toolCalls.map(({ toolCallId, toolName }) => ({
        type: 'toolCall',
        id: toolCallId,
        name: toolName,
        arguments: {},
      }));
    }
    record.message =
      descriptor.toolResult === null
        ? { role: descriptor.messageRole ?? 'unknown', content }
        : {
            role: 'toolResult',
            toolCallId: descriptor.toolResult.toolCallId,
            toolName: descriptor.toolResult.toolName,
            content: [],
          };
  }
  if (descriptor.entryType === 'compaction') {
    if (descriptor.hasRetainedTail) {
      record.retainedTail = [];
    } else {
      record.firstKeptEntryId = descriptor.firstKeptEntryId;
    }
  }
  if (descriptor.entryType === 'branch_summary') {
    record.fromId = descriptor.branchSummaryFromEntryId ?? 'root';
  }
  return record;
}

function collectTurnContextPayloadEntryIds(
  projection: LogicalSessionProjection,
  newlyEligibleEntryIds: ReadonlySet<string>,
): Set<string> {
  const descriptorsById = new Map(
    projection.entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  const childIdsById = new Map<string, string[]>();
  for (const descriptor of projection.entryDescriptors) {
    if (descriptor.parentEntryId === null) {
      continue;
    }
    const childIds = childIdsById.get(descriptor.parentEntryId) ?? [];
    childIds.push(descriptor.entryId);
    childIdsById.set(descriptor.parentEntryId, childIds);
  }
  const eligibleEntryIds = new Set(projection.eligibleContributorEntryIds);
  const payloadEntryIds = new Set(newlyEligibleEntryIds);
  const touchedUserEntryIds = new Set<string>();
  for (const entryId of newlyEligibleEntryIds) {
    let descriptor = descriptorsById.get(entryId);
    while (descriptor !== undefined) {
      if (descriptor.messageRole === 'user') {
        touchedUserEntryIds.add(descriptor.entryId);
        break;
      }
      descriptor =
        descriptor.parentEntryId === null
          ? undefined
          : descriptorsById.get(descriptor.parentEntryId);
    }
  }
  for (const userEntryId of touchedUserEntryIds) {
    const pendingEntryIds = [userEntryId];
    while (pendingEntryIds.length > 0) {
      const entryId = pendingEntryIds.shift();
      if (entryId === undefined) {
        continue;
      }
      const descriptor = descriptorsById.get(entryId);
      if (descriptor === undefined) {
        continue;
      }
      if (entryId !== userEntryId && descriptor.messageRole === 'user') {
        continue;
      }
      if (
        eligibleEntryIds.has(entryId) &&
        (descriptor.messageRole === 'user' || descriptor.messageRole === 'assistant')
      ) {
        payloadEntryIds.add(entryId);
      }
      pendingEntryIds.push(...(childIdsById.get(entryId) ?? []));
    }
  }
  return payloadEntryIds;
}

function findAppendedRecord(
  appendDelta: RecallSessionAppendDelta,
  startByte: number,
  endByte: number,
): ParsedRecallSessionRecord | undefined {
  return appendDelta.records.find(
    (record) => record.startByte === startByte && record.endByte === endByte,
  );
}

async function materializeProjectedHeader(
  options: MaterializeIncrementalRecallEligibleGraphViewOptions,
  readRange: RecallSessionSourceRangeReader,
): Promise<ParsedRecallSessionRecord> {
  const descriptor = options.logicalProjection.headerDescriptor;
  const appended = findAppendedRecord(
    options.appendDelta,
    descriptor.startByte,
    descriptor.endByte,
  );
  const record =
    appended ??
    decodeProjectedRecallRecord(
      await readExactRecallSourceRange(
        readRange,
        options.physicalProjection.sourcePath,
        descriptor.startByte,
        descriptor.endByte,
      ),
      options.physicalProjection.sourcePath,
      descriptor,
    );
  if (
    record.sourceFingerprint !== descriptor.sourceFingerprint ||
    !isSupportedRecallSessionHeader(record.value) ||
    record.value.id !== options.logicalProjection.logicalSessionId ||
    record.value.cwd !== descriptor.cwd ||
    (typeof record.value.parentSession === 'string' ? record.value.parentSession : null) !==
      descriptor.parentSessionPath
  ) {
    throw new Error(
      `Recall incremental graph header mismatch: ${options.logicalProjection.logicalSessionId}`,
    );
  }
  return record;
}

/** Materializes and strictly validates only the payload closure touched by newly eligible evidence. */
export async function materializeIncrementalRecallEligibleGraphView(
  options: MaterializeIncrementalRecallEligibleGraphViewOptions,
): Promise<IncrementalRecallEligibleGraphView> {
  const sourcePath = options.physicalProjection.sourcePath;
  const metadata = await stat(sourcePath, { bigint: true });
  if (
    options.logicalProjection.physicalSessionId !== options.physicalProjection.physicalSessionId ||
    metadata.dev.toString() !== options.physicalProjection.sourceDevice ||
    metadata.ino.toString() !== options.physicalProjection.sourceInode
  ) {
    throw new Error('Recall incremental graph source identity mismatch');
  }
  const readRange = options.readRange ?? readRecallSessionSourceRange;
  const header = await materializeProjectedHeader(options, readRange);
  const newlyEligibleEntryIds = new Set(
    options.newlyEligibleSpans.flatMap(({ contributorEntryIds }) => contributorEntryIds),
  );
  const payloadEntryIds = collectTurnContextPayloadEntryIds(
    options.logicalProjection,
    newlyEligibleEntryIds,
  );
  const canonicalRecords: CanonicalSessionRepresentation['records'] = [
    { sourceLine: header.sourceLine, value: header.value },
  ];
  for (const descriptor of options.logicalProjection.entryDescriptors) {
    let value = createProjectedEntryPlaceholder(descriptor);
    if (payloadEntryIds.has(descriptor.entryId)) {
      const appended = findAppendedRecord(
        options.appendDelta,
        descriptor.startByte,
        descriptor.endByte,
      );
      const materialized =
        appended ??
        decodeProjectedRecallRecord(
          await readExactRecallSourceRange(
            readRange,
            sourcePath,
            descriptor.startByte,
            descriptor.endByte,
          ),
          sourcePath,
          descriptor,
        );
      if (materialized.sourceFingerprint !== descriptor.sourceFingerprint) {
        throw new Error(
          `Recall incremental graph source fingerprint mismatch at ${sourcePath}:${descriptor.sourceLine}`,
        );
      }
      validateMaterializedEntryRecord(materialized, descriptor);
      value = materialized.value;
    }
    canonicalRecords.push({ sourceLine: descriptor.sourceLine, value });
  }
  const finalSourceLine =
    options.logicalProjection.entryDescriptors.at(-1)?.sourceLine ?? header.sourceLine;
  canonicalRecords.push({
    sourceLine: finalSourceLine + 1,
    value: { type: 'leaf', targetId: options.logicalProjection.effectiveLeafEntryId },
  });
  const canonicalSession: CanonicalSessionRepresentation = {
    format: SessionImportFormat.CANONICAL_JSONL,
    physicalPath: sourcePath,
    logicalSessionId: options.logicalProjection.logicalSessionId,
    sourceLineStart: header.sourceLine,
    sourceLineEnd: finalSourceLine,
    records: canonicalRecords,
  };
  const parsedGraph = parseRecallSessionGraph(canonicalSession);
  if (
    parsedGraph.currentLeafId !== options.logicalProjection.effectiveLeafEntryId ||
    parsedGraph.entries.length !== options.logicalProjection.entryDescriptors.length
  ) {
    throw new Error(
      `Recall incremental graph topology mismatch: ${options.logicalProjection.logicalSessionId}`,
    );
  }
  const graph: ParsedRecallSessionGraph = {
    ...parsedGraph,
    sessionName: options.logicalProjection.labels.at(-1) ?? parsedGraph.sessionName,
  };
  return {
    graph,
    physicalPath: sourcePath,
    logicalSessionId: options.logicalProjection.logicalSessionId,
  };
}
