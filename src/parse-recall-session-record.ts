import { createHash } from 'node:crypto';

import { SessionImportFormat } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type { RecallProjectedEntryDescriptor } from './recall-session-projection.js';

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

/** One decoded canonical JSONL record with exact physical source geometry. */
export interface ParsedRecallSessionRecord {
  sourceLine: number;
  startByte: number;
  endByte: number;
  sourceFingerprint: string;
  value: Record<string, unknown>;
}

/** Decodes one canonical record and enforces the object-plus-type boundary shared by all import paths. */
export function parseRecallSessionRecord(
  text: string,
  sourcePath: string,
  sourceLine: number,
  startByte: number,
  endByte: number,
): ParsedRecallSessionRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session JSON invalid at ${sourcePath}:${sourceLine}: ${message}`, {
      cause: error,
    });
  }
  if (!isUnknownRecord(parsed) || typeof parsed.type !== 'string') {
    throw new Error(
      `Recall session graph invalid at ${sourcePath}:${sourceLine}: each record must be an object with a type`,
    );
  }
  return {
    sourceLine,
    startByte,
    endByte,
    sourceFingerprint: createHash('sha256').update(text).digest('hex'),
    value: parsed,
  };
}

/** Reports whether a record is a complete current canonical logical-session header. */
export function isSupportedRecallSessionHeader(record: Record<string, unknown>): boolean {
  return (
    record.type === 'session' &&
    (record.version === 2 || record.version === 3) &&
    typeof record.id === 'string' &&
    record.id.length > 0 &&
    typeof record.timestamp === 'string' &&
    record.timestamp.length > 0 &&
    typeof record.cwd === 'string'
  );
}

/** Validated logical-session identity and source origin used by recall document construction. */
export interface ParsedRecallSessionHeader {
  id: string;
  cwd: string;
  parentSessionPath: string | null;
  lineIndex: number;
}

/** Validated session-graph entry retaining its complete record and physical source line. */
export interface ParsedRecallSessionEntry {
  id: string;
  parentId: string | null;
  timestamp: string;
  type: string;
  lineIndex: number;
  record: Record<string, unknown>;
}

/** Strict logical-session graph with validated topology, lifecycle state, and tool relationships. */
export interface ParsedRecallSessionGraph {
  header: ParsedRecallSessionHeader;
  entries: ParsedRecallSessionEntry[];
  entriesById: Map<string, ParsedRecallSessionEntry>;
  childEntryIdsById: Map<string, string[]>;
  currentLeafId: string | null;
  activeBranchEntryIds: Set<string>;
  activeContextEntryIds: Set<string>;
  branchPathLeafIdsByEntryId: Map<string, string[]>;
  compactedByEntryIdsByEntryId: Map<string, string[]>;
  toolCallEntryIdsByCallId: Map<string, string>;
  toolResultEntryIdsByCallId: Map<string, string>;
  sessionName: string;
}

interface ParsedSessionFileRecords {
  headers: ParsedRecallSessionHeader[];
  entries: ParsedRecallSessionEntry[];
  entriesById: Map<string, ParsedRecallSessionEntry>;
  harnessLeafTarget?: string | null;
  firstRecordLine: number;
}

interface ParsedRecallSessionHeaderRecord {
  kind: 'header';
  header: ParsedRecallSessionHeader;
}

interface ParsedSessionLeafRecord {
  kind: 'leaf';
  targetId: string | null;
}

interface ParsedRecallSessionEntryRecord {
  kind: 'entry';
  entry: ParsedRecallSessionEntry;
}

type ParsedSessionFileRecord =
  | ParsedRecallSessionHeaderRecord
  | ParsedSessionLeafRecord
  | ParsedRecallSessionEntryRecord;

function parseRequiredString(
  value: unknown,
  fieldName: string,
  sessionPath: string,
  lineIndex: number,
): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(
      `Recall session graph invalid at ${sessionPath}:${lineIndex}: ${fieldName} must be a nonempty string`,
    );
  }
  return value;
}

function readSessionEntryPath(
  entryId: string,
  entriesById: Map<string, ParsedRecallSessionEntry>,
): ParsedRecallSessionEntry[] {
  const reversed: ParsedRecallSessionEntry[] = [];
  let current = entriesById.get(entryId);
  while (current) {
    reversed.push(current);
    current = current.parentId ? entriesById.get(current.parentId) : undefined;
  }
  return reversed.reverse();
}

function findActiveContextEntryIds(
  currentLeafId: string | null,
  entriesById: Map<string, ParsedRecallSessionEntry>,
): Set<string> {
  if (!currentLeafId) {
    return new Set();
  }
  const path = readSessionEntryPath(currentLeafId, entriesById);
  let latestCompactionIndex = -1;
  for (const [index, entry] of path.entries()) {
    if (entry.type === 'compaction') {
      latestCompactionIndex = index;
    }
  }
  if (latestCompactionIndex < 0) {
    return new Set(path.map((entry) => entry.id));
  }

  const compaction = path[latestCompactionIndex];
  if (!compaction) {
    return new Set(path.map((entry) => entry.id));
  }
  const visible = new Set<string>([compaction.id]);
  if (!Array.isArray(compaction.record.retainedTail)) {
    const firstKeptEntryId = compaction.record.firstKeptEntryId;
    if (typeof firstKeptEntryId === 'string') {
      const firstKeptIndex = path.findIndex((entry) => entry.id === firstKeptEntryId);
      if (firstKeptIndex >= 0 && firstKeptIndex < latestCompactionIndex) {
        for (const entry of path.slice(firstKeptIndex, latestCompactionIndex)) {
          visible.add(entry.id);
        }
      }
    }
  }
  for (const entry of path.slice(latestCompactionIndex + 1)) {
    visible.add(entry.id);
  }
  return visible;
}

function findCompactedByEntryIds(
  entries: ParsedRecallSessionEntry[],
  entriesById: Map<string, ParsedRecallSessionEntry>,
): Map<string, string[]> {
  const compactedBy = new Map<string, string[]>();
  for (const compaction of entries.filter((entry) => entry.type === 'compaction')) {
    if (!compaction.parentId) {
      continue;
    }
    const ancestorPath = readSessionEntryPath(compaction.parentId, entriesById);
    let summarizedEnd = ancestorPath.length;
    if (!Array.isArray(compaction.record.retainedTail)) {
      const firstKeptEntryId = compaction.record.firstKeptEntryId;
      if (typeof firstKeptEntryId === 'string') {
        const firstKeptIndex = ancestorPath.findIndex((entry) => entry.id === firstKeptEntryId);
        if (firstKeptIndex >= 0) {
          summarizedEnd = firstKeptIndex;
        }
      }
    }
    for (const summarizedEntry of ancestorPath.slice(0, summarizedEnd)) {
      const compactionIds = compactedBy.get(summarizedEntry.id) ?? [];
      compactionIds.push(compaction.id);
      compactedBy.set(summarizedEntry.id, compactionIds);
    }
  }
  return compactedBy;
}

function findBranchPathLeafIds(
  entries: ParsedRecallSessionEntry[],
  entriesById: Map<string, ParsedRecallSessionEntry>,
  childEntryIdsById: Map<string, string[]>,
  currentLeafId: string | null,
): Map<string, string[]> {
  const endpoints = entries
    .filter((entry) => (childEntryIdsById.get(entry.id)?.length ?? 0) === 0)
    .map((entry) => entry.id);
  if (currentLeafId && !endpoints.includes(currentLeafId)) {
    endpoints.push(currentLeafId);
  }

  const memberships = new Map<string, string[]>();
  for (const endpoint of endpoints) {
    for (const entry of readSessionEntryPath(endpoint, entriesById)) {
      const leafIds = memberships.get(entry.id) ?? [];
      if (!leafIds.includes(endpoint)) {
        leafIds.push(endpoint);
      }
      memberships.set(entry.id, leafIds);
    }
  }
  return memberships;
}

function parseSessionFileRecord(
  parsed: unknown,
  sessionPath: string,
  lineIndex: number,
): ParsedSessionFileRecord {
  if (!isUnknownRecord(parsed) || typeof parsed.type !== 'string') {
    throw new Error(
      `Recall session graph invalid at ${sessionPath}:${lineIndex}: each line must be an object with a type`,
    );
  }
  if (parsed.type === 'session') {
    return {
      kind: 'header',
      header: {
        id: parseRequiredString(parsed.id, 'session.id', sessionPath, lineIndex),
        cwd: typeof parsed.cwd === 'string' ? parsed.cwd : '',
        parentSessionPath: typeof parsed.parentSession === 'string' ? parsed.parentSession : null,
        lineIndex,
      },
    };
  }
  if (parsed.type === 'leaf') {
    if (parsed.targetId !== null && typeof parsed.targetId !== 'string') {
      throw new Error(
        `Recall session graph invalid at ${sessionPath}:${lineIndex}: leaf.targetId must be a string or null`,
      );
    }
    return { kind: 'leaf', targetId: parsed.targetId };
  }
  const id = parseRequiredString(parsed.id, 'entry.id', sessionPath, lineIndex);
  if (parsed.parentId !== null && typeof parsed.parentId !== 'string') {
    throw new Error(
      `Recall session graph invalid at ${sessionPath}:${lineIndex}: parentId must be a string or null`,
    );
  }
  return {
    kind: 'entry',
    entry: {
      id,
      parentId: parsed.parentId,
      timestamp: parseRequiredString(parsed.timestamp, 'entry.timestamp', sessionPath, lineIndex),
      type: parsed.type,
      lineIndex,
      record: parsed,
    },
  };
}

function parseCanonicalSessionRecords(
  session: CanonicalSessionRepresentation,
  graphSource: string,
): ParsedSessionFileRecords {
  const records: ParsedSessionFileRecords = {
    headers: [],
    entries: [],
    entriesById: new Map(),
    firstRecordLine: 0,
  };
  for (const physicalRecord of session.records) {
    const record = parseSessionFileRecord(
      physicalRecord.value,
      graphSource,
      physicalRecord.sourceLine,
    );
    records.firstRecordLine ||= physicalRecord.sourceLine;
    if (record.kind === 'header') {
      records.headers.push(record.header);
    } else if (record.kind === 'leaf') {
      records.harnessLeafTarget = record.targetId;
    } else {
      if (records.entriesById.has(record.entry.id)) {
        throw new Error(
          `Recall session graph invalid at ${graphSource}:${physicalRecord.sourceLine}: duplicate entry id ${record.entry.id}`,
        );
      }
      records.entries.push(record.entry);
      records.entriesById.set(record.entry.id, record.entry);
    }
  }
  return records;
}

function validateCanonicalSessionHeader(
  headers: ParsedRecallSessionHeader[],
  firstRecordLine: number,
  sessionPath: string,
): ParsedRecallSessionHeader {
  if (headers.length !== 1) {
    throw new Error(
      `Recall session graph invalid at ${sessionPath}: expected exactly one session header, found ${headers.length}`,
    );
  }
  const header = headers[0];
  if (!header) {
    throw new Error(`Recall session graph invalid at ${sessionPath}: session header missing`);
  }
  if (header.lineIndex !== firstRecordLine) {
    throw new Error(`Recall session graph invalid at ${sessionPath}: session header must be first`);
  }
  return header;
}

function buildSessionChildEntryIds(
  entries: ParsedRecallSessionEntry[],
  entriesById: Map<string, ParsedRecallSessionEntry>,
  sessionPath: string,
): Map<string, string[]> {
  const childEntryIdsById = new Map<string, string[]>();
  for (const entry of entries) {
    if (!entry.parentId) {
      continue;
    }
    if (!entriesById.has(entry.parentId)) {
      throw new Error(
        `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: entry ${entry.id} has missing parent ${entry.parentId}`,
      );
    }
    const childIds = childEntryIdsById.get(entry.parentId) ?? [];
    childIds.push(entry.id);
    childEntryIdsById.set(entry.parentId, childIds);
  }
  return childEntryIdsById;
}

function assertSessionParentPathsAcyclic(
  entries: ParsedRecallSessionEntry[],
  entriesById: Map<string, ParsedRecallSessionEntry>,
  sessionPath: string,
): void {
  for (const entry of entries) {
    const visited = new Set<string>();
    let current: ParsedRecallSessionEntry | undefined = entry;
    while (current) {
      if (visited.has(current.id)) {
        throw new Error(
          `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: parent cycle includes ${current.id}`,
        );
      }
      visited.add(current.id);
      current = current.parentId ? entriesById.get(current.parentId) : undefined;
    }
  }
}

function resolveCurrentSessionLeafId(
  records: ParsedSessionFileRecords,
  sessionPath: string,
): string | null {
  const legacyLeafId = records.entries.at(-1)?.id ?? null;
  const currentLeafId =
    records.harnessLeafTarget === undefined ? legacyLeafId : records.harnessLeafTarget;
  if (currentLeafId && !records.entriesById.has(currentLeafId)) {
    throw new Error(
      `Recall session graph invalid at ${sessionPath}: leaf target ${currentLeafId} does not exist`,
    );
  }
  return currentLeafId;
}

function readLatestSessionName(entries: ParsedRecallSessionEntry[]): string {
  let sessionName = '';
  for (const entry of entries) {
    if (entry.type === 'session_info' && typeof entry.record.name === 'string') {
      sessionName = entry.record.name.trim();
    }
  }
  return sessionName;
}

function assertSessionCompactionAndBranchLinks(
  entries: ParsedRecallSessionEntry[],
  entriesById: Map<string, ParsedRecallSessionEntry>,
  sessionPath: string,
): void {
  for (const entry of entries) {
    if (entry.type === 'compaction' && !Array.isArray(entry.record.retainedTail)) {
      const firstKeptEntryId = entry.record.firstKeptEntryId;
      const ancestorIds = new Set(
        entry.parentId
          ? readSessionEntryPath(entry.parentId, entriesById).map((ancestor) => ancestor.id)
          : [],
      );
      if (typeof firstKeptEntryId !== 'string' || !ancestorIds.has(firstKeptEntryId)) {
        throw new Error(
          `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: compaction ${entry.id} firstKeptEntryId ${String(firstKeptEntryId)} is not an ancestor`,
        );
      }
    }
    if (entry.type === 'branch_summary') {
      const fromId = entry.record.fromId;
      if (fromId !== 'root' && (typeof fromId !== 'string' || !entriesById.has(fromId))) {
        throw new Error(
          `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: branch summary ${entry.id} fromId ${String(fromId)} does not name an entry or root`,
        );
      }
    }
  }
}

function isBlankToolPlaceholder(toolCallId: unknown, toolName: unknown): boolean {
  return toolCallId === '' && toolName === '';
}

function findSessionToolEntryLinks(
  entries: ParsedRecallSessionEntry[],
  sessionPath: string,
): {
  toolCallEntryIdsByCallId: Map<string, string>;
  toolResultEntryIdsByCallId: Map<string, string>;
} {
  const toolCallEntryIdsByCallId = new Map<string, string>();
  const toolCallNamesByCallId = new Map<string, string>();
  const toolResultEntryIdsByCallId = new Map<string, string>();
  const toolResultNamesByCallId = new Map<string, string>();
  for (const entry of entries) {
    if (entry.type !== 'message' || !isUnknownRecord(entry.record.message)) {
      continue;
    }
    const message = entry.record.message;
    if (message.role === 'assistant' && Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isUnknownRecord(block) || block.type !== 'toolCall') {
          continue;
        }
        if (isBlankToolPlaceholder(block.id, block.name)) {
          continue;
        }
        const toolCallId = parseRequiredString(
          block.id,
          'toolCall.id',
          sessionPath,
          entry.lineIndex,
        );
        const toolName = parseRequiredString(
          block.name,
          'toolCall.name',
          sessionPath,
          entry.lineIndex,
        );
        if (toolCallEntryIdsByCallId.has(toolCallId)) {
          throw new Error(
            `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: duplicate tool call id ${toolCallId}`,
          );
        }
        toolCallEntryIdsByCallId.set(toolCallId, entry.id);
        toolCallNamesByCallId.set(toolCallId, toolName);
      }
    }
    if (message.role === 'toolResult') {
      if (isBlankToolPlaceholder(message.toolCallId, message.toolName)) {
        continue;
      }
      const toolCallId = parseRequiredString(
        message.toolCallId,
        'toolResult.toolCallId',
        sessionPath,
        entry.lineIndex,
      );
      const toolName = parseRequiredString(
        message.toolName,
        'toolResult.toolName',
        sessionPath,
        entry.lineIndex,
      );
      if (toolResultEntryIdsByCallId.has(toolCallId)) {
        throw new Error(
          `Recall session graph invalid at ${sessionPath}:${entry.lineIndex}: duplicate tool result id ${toolCallId}`,
        );
      }
      toolResultEntryIdsByCallId.set(toolCallId, entry.id);
      toolResultNamesByCallId.set(toolCallId, toolName);
    }
  }
  for (const [toolCallId, resultEntryId] of toolResultEntryIdsByCallId) {
    const callEntryId = toolCallEntryIdsByCallId.get(toolCallId);
    if (!callEntryId) {
      throw new Error(
        `Recall session graph invalid at ${sessionPath}: tool result ${toolCallId} has no matching tool call`,
      );
    }
    const callToolName = toolCallNamesByCallId.get(toolCallId);
    const resultToolName = toolResultNamesByCallId.get(toolCallId);
    if (callToolName !== resultToolName) {
      throw new Error(
        `Recall session graph invalid at ${sessionPath}: tool result ${toolCallId} names ${String(resultToolName)}, but call names ${String(callToolName)} (entries ${callEntryId} and ${resultEntryId})`,
      );
    }
  }
  return { toolCallEntryIdsByCallId, toolResultEntryIdsByCallId };
}

/** Reads one root-to-entry path from validated scalar descriptors, rejecting missing links and cycles. */
export function readProjectedRecallSessionEntryPath(
  entryId: string,
  entriesById: ReadonlyMap<string, RecallProjectedEntryDescriptor>,
): RecallProjectedEntryDescriptor[] | null {
  const reversed: RecallProjectedEntryDescriptor[] = [];
  const visited = new Set<string>();
  let current = entriesById.get(entryId);
  if (current === undefined) {
    return null;
  }
  while (current !== undefined) {
    if (visited.has(current.entryId)) {
      return null;
    }
    visited.add(current.entryId);
    reversed.push(current);
    if (current.parentEntryId === null) {
      return reversed.reverse();
    }
    current = entriesById.get(current.parentEntryId);
  }
  return null;
}

/** Validates persisted scalar parent, compaction, and tool links without conversation payloads. */
export function validateProjectedRecallSessionEntryLinks(
  entryDescriptors: readonly RecallProjectedEntryDescriptor[],
): boolean {
  const entriesById = new Map(
    entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  if (entriesById.size !== entryDescriptors.length) {
    return false;
  }
  const toolCalls = new Map<string, string>();
  const toolResults = new Set<string>();
  for (const descriptor of entryDescriptors) {
    if (descriptor.parentEntryId !== null && !entriesById.has(descriptor.parentEntryId)) {
      return false;
    }
    if (readProjectedRecallSessionEntryPath(descriptor.entryId, entriesById) === null) {
      return false;
    }
    if (descriptor.entryType === 'compaction' && !descriptor.hasRetainedTail) {
      const parentPath =
        descriptor.parentEntryId === null
          ? []
          : readProjectedRecallSessionEntryPath(descriptor.parentEntryId, entriesById);
      if (
        parentPath === null ||
        descriptor.firstKeptEntryId === null ||
        !parentPath.some(({ entryId }) => entryId === descriptor.firstKeptEntryId)
      ) {
        return false;
      }
    }
    for (const toolCall of descriptor.toolCalls) {
      if (toolCalls.has(toolCall.toolCallId)) {
        return false;
      }
      toolCalls.set(toolCall.toolCallId, toolCall.toolName);
    }
    if (descriptor.toolResult !== null) {
      if (toolResults.has(descriptor.toolResult.toolCallId)) {
        return false;
      }
      toolResults.add(descriptor.toolResult.toolCallId);
    }
  }
  return entryDescriptors.every(
    ({ toolResult }) =>
      toolResult === null || toolCalls.get(toolResult.toolCallId) === toolResult.toolName,
  );
}

/** Strictly validates one canonical logical session graph and all cross-record links. */
export function parseRecallSessionGraph(
  session: CanonicalSessionRepresentation,
): ParsedRecallSessionGraph {
  const graphSource =
    session.format === SessionImportFormat.PI_SESSION_REUSE_HISTORY
      ? `${session.physicalPath} (logical session ${session.logicalSessionId})`
      : session.physicalPath;
  const records = parseCanonicalSessionRecords(session, graphSource);
  const header = validateCanonicalSessionHeader(
    records.headers,
    records.firstRecordLine,
    graphSource,
  );
  const childEntryIdsById = buildSessionChildEntryIds(
    records.entries,
    records.entriesById,
    graphSource,
  );
  assertSessionParentPathsAcyclic(records.entries, records.entriesById, graphSource);
  assertSessionCompactionAndBranchLinks(records.entries, records.entriesById, graphSource);
  const currentLeafId = resolveCurrentSessionLeafId(records, graphSource);
  const activeBranchEntryIds = new Set(
    currentLeafId
      ? readSessionEntryPath(currentLeafId, records.entriesById).map((entry) => entry.id)
      : [],
  );
  const toolEntryLinks = findSessionToolEntryLinks(records.entries, graphSource);

  return {
    header,
    entries: records.entries,
    entriesById: records.entriesById,
    childEntryIdsById,
    currentLeafId,
    activeBranchEntryIds,
    activeContextEntryIds: findActiveContextEntryIds(currentLeafId, records.entriesById),
    branchPathLeafIdsByEntryId: findBranchPathLeafIds(
      records.entries,
      records.entriesById,
      childEntryIdsById,
      currentLeafId,
    ),
    compactedByEntryIdsByEntryId: findCompactedByEntryIds(records.entries, records.entriesById),
    ...toolEntryLinks,
    sessionName: readLatestSessionName(records.entries),
  };
}
