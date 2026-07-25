import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

/** Version of the source and graph provenance stored on recall evidence documents. */
export const SESSION_CONVERSATION_SCHEMA_VERSION = 4;

/** A Pi session ID that cannot be passed where a session entry ID is required. */
export interface PiSessionId {
  readonly value: string;
}

/** A Pi session entry ID that cannot be passed where a session ID is required. */
export interface PiSessionEntryId {
  readonly value: string;
}

/** Exact token IDs produced by the configured embedding-model tokenizer. */
export interface ConversationTextEncoding {
  readonly ids: readonly number[];
}

/** Encodes conversation text locally with the exact embedding-model tokenizer. */
export interface ConversationTextTokenizer {
  encodeConversationText(text: string): ConversationTextEncoding;
}

/** Token limits for one atomic conversation chunk; production limits cannot be exceeded. */
export interface SessionConversationChunkOptions {
  tokenizer: ConversationTextTokenizer;
  maxTokens?: number;
  overlapTokens?: number;
}

/** A token-bounded recall evidence document with complete source and session-graph provenance. */
export interface SessionConversationChunk {
  schemaVersion: number;
  documentKind: 'conversation' | 'turn_context' | 'summary' | 'tool';
  summaryKind: 'compaction' | 'branch' | null;
  evidenceKind:
    | 'conversation'
    | 'turn_context'
    | 'compaction_summary'
    | 'branch_summary'
    | 'tool_call'
    | 'tool_result'
    | 'bash_execution';
  evidencePart: 'content' | 'name' | 'arguments' | 'result' | 'command' | 'output';
  isDenseSearchable: boolean;
  id: string;
  checksum: string;
  sessionId: PiSessionId;
  sessionPath: string;
  parentSessionPath: string | null;
  cwd: string;
  projectPath: string;
  sessionName: string;
  entryId: PiSessionEntryId;
  parentEntryId: PiSessionEntryId | null;
  childEntryIds: PiSessionEntryId[];
  contributingEntryIds: PiSessionEntryId[];
  currentLeafId: PiSessionEntryId | null;
  branchPathLeafIds: PiSessionEntryId[];
  isOnActiveBranch: boolean;
  isVisibleInActiveContext: boolean;
  compactedByEntryIds: PiSessionEntryId[];
  compactionFirstKeptEntryId: PiSessionEntryId | null;
  branchSummaryFromEntryId: PiSessionEntryId | null;
  role: 'user' | 'assistant' | 'turn' | 'summary' | 'custom' | 'tool';
  timestamp: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  sourceBlockStart: number | null;
  sourceBlockEnd: number | null;
  characterStart: number;
  characterEnd: number;
  tokenStart: number;
  tokenEnd: number;
  tokenCount: number;
  overlapTokenCount: number;
  textRunId: string;
  textRunIndex: number;
  chunkIndex: number;
  chunkCount: number;
  siblingIds: string[];
  previousSiblingId: string | null;
  nextSiblingId: string | null;
  toolCallId: string | null;
  toolName: string | null;
  toolCallEntryId: PiSessionEntryId | null;
  toolResultEntryId: PiSessionEntryId | null;
  toolError: boolean | null;
  content: string;
}

interface ParsedSessionHeader {
  id: string;
  cwd: string;
  parentSessionPath: string | null;
  lineIndex: number;
}

interface ParsedSessionEntry {
  id: string;
  parentId: string | null;
  timestamp: string;
  type: string;
  lineIndex: number;
  record: Record<string, unknown>;
}

interface ParsedSessionGraph {
  header: ParsedSessionHeader;
  entries: ParsedSessionEntry[];
  entriesById: Map<string, ParsedSessionEntry>;
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
  headers: ParsedSessionHeader[];
  entries: ParsedSessionEntry[];
  entriesById: Map<string, ParsedSessionEntry>;
  harnessLeafTarget: string | null | undefined;
  firstRecordLine: number;
}

interface ParsedSessionHeaderRecord {
  kind: 'header';
  header: ParsedSessionHeader;
}

interface ParsedSessionLeafRecord {
  kind: 'leaf';
  targetId: string | null;
}

interface ParsedSessionEntryRecord {
  kind: 'entry';
  entry: ParsedSessionEntry;
}

type ParsedSessionFileRecord =
  | ParsedSessionHeaderRecord
  | ParsedSessionLeafRecord
  | ParsedSessionEntryRecord;

interface VisibleConversationTextRun {
  text: string;
  textRunIndex: number;
  sourceBlockStart: number | null;
  sourceBlockEnd: number | null;
}

interface ConversationChunkSpan {
  characterStart: number;
  characterEnd: number;
  tokenStart: number;
  tokenEnd: number;
  tokenCount: number;
  overlapTokenCount: number;
  content: string;
}

interface TurnContextText {
  userText: string;
  assistantText: string;
}

interface PendingConversationDocumentBase {
  entry: ParsedSessionEntry;
  contributingEntries: ParsedSessionEntry[];
  role: SessionConversationChunk['role'];
  summaryKind: SessionConversationChunk['summaryKind'];
  evidenceKind: SessionConversationChunk['evidenceKind'];
  evidencePart: SessionConversationChunk['evidencePart'];
  isDenseSearchable: boolean;
  compactionFirstKeptEntryId: string | null;
  branchSummaryFromEntryId: string | null;
  toolCallId: string | null;
  toolName: string | null;
  toolCallEntryId: string | null;
  toolResultEntryId: string | null;
  toolError: boolean | null;
  preserveVerbatim: boolean;
  textRun: VisibleConversationTextRun;
}

interface PendingEntryScopedDocument extends PendingConversationDocumentBase {
  documentKind: 'conversation' | 'summary' | 'tool';
}

interface PendingTurnContextDocument extends PendingConversationDocumentBase {
  documentKind: 'turn_context';
  turnContextText: TurnContextText;
}

type PendingConversationDocument = PendingEntryScopedDocument | PendingTurnContextDocument;

interface PendingTurnContextPath {
  entry: ParsedSessionEntry;
  contributingEntries: ParsedSessionEntry[];
  assistantTexts: string[];
}

interface SessionConversationChunkContext {
  graph: ParsedSessionGraph;
  sessionPath: string;
  tokenizer: ConversationTextTokenizer;
  maxTokens: number;
  overlapTokens: number;
  sessionId: PiSessionId;
  currentLeafId: PiSessionEntryId | null;
}

function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hashConversationValue(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function createPiSessionEntryId(value: string): PiSessionEntryId {
  return { value };
}

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
  entriesById: Map<string, ParsedSessionEntry>,
): ParsedSessionEntry[] {
  const reversed: ParsedSessionEntry[] = [];
  let current = entriesById.get(entryId);
  while (current) {
    reversed.push(current);
    current = current.parentId ? entriesById.get(current.parentId) : undefined;
  }
  return reversed.reverse();
}

function findActiveContextEntryIds(
  currentLeafId: string | null,
  entriesById: Map<string, ParsedSessionEntry>,
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
  entries: ParsedSessionEntry[],
  entriesById: Map<string, ParsedSessionEntry>,
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
  entries: ParsedSessionEntry[],
  entriesById: Map<string, ParsedSessionEntry>,
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
  line: string,
  sessionPath: string,
  lineIndex: number,
): ParsedSessionFileRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session JSON invalid at ${sessionPath}:${lineIndex}: ${message}`, {
      cause: error,
    });
  }
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

async function readSessionFileRecords(sessionPath: string): Promise<ParsedSessionFileRecords> {
  const records: ParsedSessionFileRecords = {
    headers: [],
    entries: [],
    entriesById: new Map(),
    harnessLeafTarget: undefined,
    firstRecordLine: 0,
  };
  const lines = createInterface({
    input: createReadStream(sessionPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  let lineIndex = 0;
  for await (const line of lines) {
    lineIndex += 1;
    if (!line.trim()) {
      continue;
    }
    const record = parseSessionFileRecord(line, sessionPath, lineIndex);
    records.firstRecordLine ||= lineIndex;
    if (record.kind === 'header') {
      records.headers.push(record.header);
    } else if (record.kind === 'leaf') {
      records.harnessLeafTarget = record.targetId;
    } else {
      if (records.entriesById.has(record.entry.id)) {
        throw new Error(
          `Recall session graph invalid at ${sessionPath}:${lineIndex}: duplicate entry id ${record.entry.id}`,
        );
      }
      records.entries.push(record.entry);
      records.entriesById.set(record.entry.id, record.entry);
    }
  }
  return records;
}

function readSingleSessionHeader(
  headers: ParsedSessionHeader[],
  firstRecordLine: number,
  sessionPath: string,
): ParsedSessionHeader {
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
  entries: ParsedSessionEntry[],
  entriesById: Map<string, ParsedSessionEntry>,
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
  entries: ParsedSessionEntry[],
  entriesById: Map<string, ParsedSessionEntry>,
  sessionPath: string,
): void {
  for (const entry of entries) {
    const visited = new Set<string>();
    let current: ParsedSessionEntry | undefined = entry;
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

function readLatestSessionName(entries: ParsedSessionEntry[]): string {
  let sessionName = '';
  for (const entry of entries) {
    if (entry.type === 'session_info' && typeof entry.record.name === 'string') {
      sessionName = entry.record.name.trim();
    }
  }
  return sessionName;
}

function findSessionToolEntryLinks(entries: ParsedSessionEntry[]): {
  toolCallEntryIdsByCallId: Map<string, string>;
  toolResultEntryIdsByCallId: Map<string, string>;
} {
  const toolCallEntryIdsByCallId = new Map<string, string>();
  const toolResultEntryIdsByCallId = new Map<string, string>();
  for (const entry of entries) {
    if (entry.type !== 'message' || !isUnknownRecord(entry.record.message)) {
      continue;
    }
    const message = entry.record.message;
    if (message.role === 'assistant' && Array.isArray(message.content)) {
      for (const block of message.content) {
        if (
          isUnknownRecord(block) &&
          block.type === 'toolCall' &&
          typeof block.id === 'string' &&
          block.id
        ) {
          toolCallEntryIdsByCallId.set(block.id, entry.id);
        }
      }
    }
    if (
      message.role === 'toolResult' &&
      typeof message.toolCallId === 'string' &&
      message.toolCallId
    ) {
      toolResultEntryIdsByCallId.set(message.toolCallId, entry.id);
    }
  }
  return { toolCallEntryIdsByCallId, toolResultEntryIdsByCallId };
}

async function readValidatedSessionGraph(sessionPath: string): Promise<ParsedSessionGraph> {
  const records = await readSessionFileRecords(sessionPath);
  const header = readSingleSessionHeader(records.headers, records.firstRecordLine, sessionPath);
  const childEntryIdsById = buildSessionChildEntryIds(
    records.entries,
    records.entriesById,
    sessionPath,
  );
  assertSessionParentPathsAcyclic(records.entries, records.entriesById, sessionPath);
  const currentLeafId = resolveCurrentSessionLeafId(records, sessionPath);
  const activeBranchEntryIds = new Set(
    currentLeafId
      ? readSessionEntryPath(currentLeafId, records.entriesById).map((entry) => entry.id)
      : [],
  );
  const toolEntryLinks = findSessionToolEntryLinks(records.entries);

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

function extractVisibleConversationTextRuns(content: unknown): VisibleConversationTextRun[] {
  if (typeof content === 'string') {
    const text = content.trim();
    return text ? [{ text, textRunIndex: 0, sourceBlockStart: null, sourceBlockEnd: null }] : [];
  }
  if (!Array.isArray(content)) {
    return [];
  }

  const runs: VisibleConversationTextRun[] = [];
  let adjacentText: string[] = [];
  let sourceBlockStart: number | null = null;
  let sourceBlockEnd: number | null = null;
  const finishTextRun = (): void => {
    const text = adjacentText.join('\n').trim();
    if (text) {
      runs.push({
        text,
        textRunIndex: runs.length,
        sourceBlockStart,
        sourceBlockEnd,
      });
    }
    adjacentText = [];
    sourceBlockStart = null;
    sourceBlockEnd = null;
  };

  for (const [blockIndex, block] of content.entries()) {
    if (
      isUnknownRecord(block) &&
      block.type === 'text' &&
      typeof block.text === 'string' &&
      block.text.trim()
    ) {
      sourceBlockStart ??= blockIndex;
      sourceBlockEnd = blockIndex;
      adjacentText.push(block.text);
      continue;
    }
    finishTextRun();
  }
  finishTextRun();
  return runs;
}

function createUnicodeCharacterBoundaries(text: string): number[] {
  const boundaries = [0];
  let offset = 0;
  for (const character of text) {
    offset += character.length;
    boundaries.push(offset);
  }
  return boundaries;
}

function findBoundaryIndex(boundaries: number[], value: number): number {
  let low = 0;
  let high = boundaries.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = boundaries[middle];
    if (candidate === value) {
      return middle;
    }
    if (candidate !== undefined && candidate < value) {
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return Math.max(0, high);
}

function countConversationTokens(text: string, tokenizer: ConversationTextTokenizer): number {
  return tokenizer.encodeConversationText(text).ids.length;
}

function findTokenLimitedTextEnd(
  text: string,
  start: number,
  boundaries: number[],
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): number {
  const startIndex = findBoundaryIndex(boundaries, start);
  let acceptedIndex = startIndex;
  let rejectedIndex = boundaries.length;
  let step = 1;
  while (startIndex + step < boundaries.length) {
    const candidateIndex = startIndex + step;
    const candidate = boundaries[candidateIndex];
    if (
      candidate === undefined ||
      countConversationTokens(text.slice(start, candidate).trim(), tokenizer) > maxTokens
    ) {
      rejectedIndex = candidateIndex;
      break;
    }
    acceptedIndex = candidateIndex;
    step *= 2;
  }
  if (rejectedIndex === boundaries.length) {
    const finalIndex = boundaries.length - 1;
    const finalBoundary = boundaries[finalIndex];
    if (
      finalBoundary !== undefined &&
      countConversationTokens(text.slice(start, finalBoundary).trim(), tokenizer) <= maxTokens
    ) {
      acceptedIndex = finalIndex;
    } else {
      rejectedIndex = finalIndex;
    }
  }

  let low = acceptedIndex + 1;
  let high = rejectedIndex - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = boundaries[middle];
    if (candidate === undefined) {
      high = middle - 1;
      continue;
    }
    if (countConversationTokens(text.slice(start, candidate).trim(), tokenizer) <= maxTokens) {
      acceptedIndex = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  const accepted = boundaries[acceptedIndex] ?? start;
  if (accepted === start) {
    throw new Error(
      `Recall chunk policy cannot fit one Unicode character within maxTokens=${maxTokens}`,
    );
  }
  return accepted;
}

function collectNaturalBoundaryCandidates(
  text: string,
  start: number,
  hardEnd: number,
  pattern: RegExp,
  resolveBoundary: (match: RegExpExecArray) => number,
): number[] {
  const segment = text.slice(start, hardEnd);
  const candidates: number[] = [];
  pattern.lastIndex = 0;
  let match = pattern.exec(segment);
  while (match) {
    candidates.push(start + resolveBoundary(match));
    if (match[0].length === 0) {
      pattern.lastIndex += 1;
    }
    match = pattern.exec(segment);
  }
  return candidates;
}

function findNaturalConversationBoundary(text: string, start: number, hardEnd: number): number {
  const minimum = start + Math.floor((hardEnd - start) / 2);
  const boundaryGroups = [
    collectNaturalBoundaryCandidates(
      text,
      start,
      hardEnd,
      /^#{1,6}[ \t]+/gmu,
      (match) => match.index,
    ),
    collectNaturalBoundaryCandidates(
      text,
      start,
      hardEnd,
      /\n[ \t]*\n+/gu,
      (match) => match.index + match[0].length,
    ),
    collectNaturalBoundaryCandidates(
      text,
      start,
      hardEnd,
      /^(?:```|~~~).*$/gmu,
      (match) => match.index + match[0].length,
    ),
    collectNaturalBoundaryCandidates(
      text,
      start,
      hardEnd,
      /\n/gu,
      (match) => match.index + match[0].length,
    ),
    collectNaturalBoundaryCandidates(
      text,
      start,
      hardEnd,
      /[.!?](?:["')\]]*)[ \t]+/gu,
      (match) => match.index + match[0].trimEnd().length,
    ),
  ];
  for (const candidates of boundaryGroups) {
    const boundary = candidates.filter((candidate) => candidate > minimum).at(-1);
    if (boundary !== undefined) {
      return boundary;
    }
  }
  return hardEnd;
}

function trimConversationSpan(
  text: string,
  characterStart: number,
  characterEnd: number,
): { characterStart: number; characterEnd: number } {
  let start = characterStart;
  let end = characterEnd;
  while (start < end && /\s/u.test(text[start] ?? '')) {
    start += 1;
  }
  while (end > start && /\s/u.test(text[end - 1] ?? '')) {
    end -= 1;
  }
  return { characterStart: start, characterEnd: end };
}

function findConversationOverlapStart(
  text: string,
  chunkStart: number,
  chunkEnd: number,
  boundaries: number[],
  tokenizer: ConversationTextTokenizer,
  overlapTokens: number,
): number {
  if (overlapTokens === 0) {
    return chunkEnd;
  }
  let low = findBoundaryIndex(boundaries, chunkStart);
  let high = findBoundaryIndex(boundaries, chunkEnd);
  let accepted = chunkEnd;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = boundaries[middle];
    if (candidate === undefined) {
      low = middle + 1;
      continue;
    }
    if (
      countConversationTokens(text.slice(candidate, chunkEnd).trim(), tokenizer) <= overlapTokens
    ) {
      accepted = candidate;
      high = middle - 1;
    } else {
      low = middle + 1;
    }
  }
  return trimConversationSpan(text, accepted, chunkEnd).characterStart;
}

function splitConversationTextByTokens(
  text: string,
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
  overlapTokens: number,
): ConversationChunkSpan[] {
  if (!text) {
    return [];
  }
  const boundaries = createUnicodeCharacterBoundaries(text);
  const spans: ConversationChunkSpan[] = [];
  let start = 0;
  let previousEnd = 0;

  while (start < text.length) {
    const hardEnd = findTokenLimitedTextEnd(text, start, boundaries, tokenizer, maxTokens);
    const naturalEnd =
      hardEnd < text.length ? findNaturalConversationBoundary(text, start, hardEnd) : hardEnd;
    const trimmed = trimConversationSpan(text, start, naturalEnd);
    const content = text.slice(trimmed.characterStart, trimmed.characterEnd);
    const tokenCount = countConversationTokens(content, tokenizer);
    if (tokenCount > maxTokens) {
      throw new Error(
        `Recall chunk policy emitted ${tokenCount} tokens, exceeding maxTokens=${maxTokens}`,
      );
    }
    const overlapText =
      spans.length > 0 && previousEnd > trimmed.characterStart
        ? text.slice(trimmed.characterStart, previousEnd)
        : '';
    const overlapTokenCount = countConversationTokens(overlapText, tokenizer);
    if (overlapTokenCount > overlapTokens) {
      throw new Error(
        `Recall chunk policy emitted ${overlapTokenCount} overlap tokens, exceeding overlapTokens=${overlapTokens}`,
      );
    }
    if (content) {
      const previousTokenEnd = spans.at(-1)?.tokenEnd ?? 0;
      const tokenStart = Math.max(0, previousTokenEnd - overlapTokenCount);
      spans.push({
        characterStart: trimmed.characterStart,
        characterEnd: trimmed.characterEnd,
        tokenStart,
        tokenEnd: tokenStart + tokenCount,
        tokenCount,
        overlapTokenCount,
        content,
      });
    }
    if (naturalEnd >= text.length) {
      break;
    }
    previousEnd = trimmed.characterEnd;
    const nextStart = findConversationOverlapStart(
      text,
      trimmed.characterStart,
      trimmed.characterEnd,
      boundaries,
      tokenizer,
      overlapTokens,
    );
    start = nextStart > start ? nextStart : naturalEnd;
  }
  return spans;
}

function findTokenLimitedVerbatimEnd(
  text: string,
  start: number,
  boundaries: number[],
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): number {
  const startIndex = findBoundaryIndex(boundaries, start);
  let low = startIndex + 1;
  let high = boundaries.length - 1;
  let acceptedIndex = startIndex;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = boundaries[middle];
    if (candidate === undefined) {
      high = middle - 1;
      continue;
    }
    if (countConversationTokens(text.slice(start, candidate), tokenizer) <= maxTokens) {
      acceptedIndex = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  const accepted = boundaries[acceptedIndex] ?? start;
  if (accepted === start) {
    throw new Error(
      `Recall tool evidence cannot fit one Unicode character within maxTokens=${maxTokens}`,
    );
  }
  return accepted;
}

function splitVerbatimToolEvidenceByTokens(
  text: string,
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): ConversationChunkSpan[] {
  if (!text || !text.trim()) {
    return [];
  }
  const boundaries = createUnicodeCharacterBoundaries(text);
  const spans: ConversationChunkSpan[] = [];
  let characterStart = 0;
  let tokenStart = 0;
  while (characterStart < text.length) {
    const hardEnd = findTokenLimitedVerbatimEnd(
      text,
      characterStart,
      boundaries,
      tokenizer,
      maxTokens,
    );
    const characterEnd =
      hardEnd < text.length
        ? findNaturalConversationBoundary(text, characterStart, hardEnd)
        : hardEnd;
    const content = text.slice(characterStart, characterEnd);
    const tokenCount = countConversationTokens(content, tokenizer);
    if (tokenCount > maxTokens) {
      throw new Error(
        `Recall tool evidence emitted ${tokenCount} tokens, exceeding maxTokens=${maxTokens}`,
      );
    }
    spans.push({
      characterStart,
      characterEnd,
      tokenStart,
      tokenEnd: tokenStart + tokenCount,
      tokenCount,
      overlapTokenCount: 0,
      content,
    });
    characterStart = characterEnd;
    tokenStart += tokenCount;
  }
  return spans;
}

function createConversationTextDocuments(
  entry: ParsedSessionEntry,
  role: 'user' | 'assistant' | 'custom',
  content: unknown,
): PendingConversationDocument[] {
  return extractVisibleConversationTextRuns(content).map((textRun) => ({
    entry,
    contributingEntries: [entry],
    role,
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    isDenseSearchable: true,
    compactionFirstKeptEntryId: null,
    branchSummaryFromEntryId: null,
    toolCallId: null,
    toolName: null,
    toolCallEntryId: null,
    toolResultEntryId: null,
    toolError: null,
    preserveVerbatim: false,
    textRun,
  }));
}

function createSummaryTextDocument(
  entry: ParsedSessionEntry,
  summaryKind: 'compaction' | 'branch',
): PendingConversationDocument[] {
  const text = typeof entry.record.summary === 'string' ? entry.record.summary.trim() : '';
  if (!text) {
    return [];
  }
  return [
    {
      entry,
      contributingEntries: [entry],
      role: 'summary',
      documentKind: 'summary',
      summaryKind,
      evidenceKind: summaryKind === 'compaction' ? 'compaction_summary' : 'branch_summary',
      evidencePart: 'content',
      isDenseSearchable: true,
      compactionFirstKeptEntryId:
        summaryKind === 'compaction' && typeof entry.record.firstKeptEntryId === 'string'
          ? entry.record.firstKeptEntryId
          : null,
      branchSummaryFromEntryId:
        summaryKind === 'branch' && typeof entry.record.fromId === 'string'
          ? entry.record.fromId
          : null,
      toolCallId: null,
      toolName: null,
      toolCallEntryId: null,
      toolResultEntryId: null,
      toolError: null,
      preserveVerbatim: false,
      textRun: {
        text,
        textRunIndex: 0,
        sourceBlockStart: null,
        sourceBlockEnd: null,
      },
    },
  ];
}

function createToolCallDocuments(
  graph: ParsedSessionGraph,
  entry: ParsedSessionEntry,
  content: unknown,
): PendingConversationDocument[] {
  if (!Array.isArray(content)) {
    return [];
  }
  return content.flatMap((block, blockIndex) => {
    if (
      !isUnknownRecord(block) ||
      block.type !== 'toolCall' ||
      typeof block.id !== 'string' ||
      !block.id ||
      typeof block.name !== 'string' ||
      !block.name
    ) {
      return [];
    }
    const serializedArguments = JSON.stringify(block.arguments);
    const resultEntryId = graph.toolResultEntryIdsByCallId.get(block.id) ?? null;
    const baseDocument = {
      entry,
      contributingEntries: [entry],
      role: 'tool' as const,
      documentKind: 'tool' as const,
      summaryKind: null,
      evidenceKind: 'tool_call' as const,
      isDenseSearchable: false,
      compactionFirstKeptEntryId: null,
      branchSummaryFromEntryId: null,
      toolCallId: block.id,
      toolName: block.name,
      toolCallEntryId: entry.id,
      toolResultEntryId: resultEntryId,
      toolError: null,
      preserveVerbatim: true,
    };
    const documents: PendingConversationDocument[] = [
      {
        ...baseDocument,
        evidencePart: 'name',
        textRun: {
          text: block.name,
          textRunIndex: blockIndex * 2,
          sourceBlockStart: blockIndex,
          sourceBlockEnd: blockIndex,
        },
      },
    ];
    if (serializedArguments !== undefined) {
      documents.push({
        ...baseDocument,
        evidencePart: 'arguments',
        textRun: {
          text: serializedArguments,
          textRunIndex: blockIndex * 2 + 1,
          sourceBlockStart: blockIndex,
          sourceBlockEnd: blockIndex,
        },
      });
    }
    return documents;
  });
}

function createToolResultDocuments(
  graph: ParsedSessionGraph,
  entry: ParsedSessionEntry,
  message: Record<string, unknown>,
): PendingConversationDocument[] {
  if (
    typeof message.toolCallId !== 'string' ||
    !message.toolCallId ||
    typeof message.toolName !== 'string' ||
    !message.toolName ||
    !Array.isArray(message.content)
  ) {
    return [];
  }
  const toolCallId = message.toolCallId;
  const toolName = message.toolName;
  const callEntryId = graph.toolCallEntryIdsByCallId.get(toolCallId) ?? null;
  return message.content.flatMap((block, blockIndex) => {
    if (
      !isUnknownRecord(block) ||
      block.type !== 'text' ||
      typeof block.text !== 'string' ||
      !block.text.trim()
    ) {
      return [];
    }
    return [
      {
        entry,
        contributingEntries: [entry],
        role: 'tool',
        documentKind: 'tool',
        summaryKind: null,
        evidenceKind: 'tool_result',
        evidencePart: 'result',
        isDenseSearchable: false,
        compactionFirstKeptEntryId: null,
        branchSummaryFromEntryId: null,
        toolCallId,
        toolName,
        toolCallEntryId: callEntryId,
        toolResultEntryId: entry.id,
        toolError: typeof message.isError === 'boolean' ? message.isError : null,
        preserveVerbatim: true,
        textRun: {
          text: block.text,
          textRunIndex: blockIndex,
          sourceBlockStart: blockIndex,
          sourceBlockEnd: blockIndex,
        },
      } satisfies PendingConversationDocument,
    ];
  });
}

function createBashExecutionDocuments(
  entry: ParsedSessionEntry,
  message: Record<string, unknown>,
): PendingConversationDocument[] {
  let isError: boolean | null = null;
  if (message.cancelled === true) {
    isError = true;
  } else if (typeof message.exitCode === 'number') {
    isError = message.exitCode !== 0;
  }
  const evidenceParts = [
    { evidencePart: 'command' as const, text: message.command, textRunIndex: 0 },
    { evidencePart: 'output' as const, text: message.output, textRunIndex: 1 },
  ];
  return evidenceParts.flatMap(({ evidencePart, text, textRunIndex }) => {
    if (typeof text !== 'string' || !text.trim()) {
      return [];
    }
    return [
      {
        entry,
        contributingEntries: [entry],
        role: 'tool',
        documentKind: 'tool',
        summaryKind: null,
        evidenceKind: 'bash_execution',
        evidencePart,
        isDenseSearchable: false,
        compactionFirstKeptEntryId: null,
        branchSummaryFromEntryId: null,
        toolCallId: null,
        toolName: 'bash',
        toolCallEntryId: null,
        toolResultEntryId: null,
        toolError: isError,
        preserveVerbatim: true,
        textRun: {
          text,
          textRunIndex,
          sourceBlockStart: null,
          sourceBlockEnd: null,
        },
      } satisfies PendingConversationDocument,
    ];
  });
}

function readSessionGraphEntry(graph: ParsedSessionGraph, entryId: string): ParsedSessionEntry {
  const entry = graph.entriesById.get(entryId);
  if (!entry) {
    throw new Error(`Recall turn context entry missing from graph: ${entryId}`);
  }
  return entry;
}

function readSessionEntryMessage(entry: ParsedSessionEntry): Record<string, unknown> | null {
  if (entry.type !== 'message' || !isUnknownRecord(entry.record.message)) {
    return null;
  }
  return entry.record.message;
}

function readTurnContextRoleText(entry: ParsedSessionEntry, role: 'user' | 'assistant'): string {
  const message = readSessionEntryMessage(entry);
  if (message?.role !== role) {
    return '';
  }
  return extractVisibleConversationTextRuns(message.content)
    .map((textRun) => textRun.text)
    .join('\n\n');
}

function createContributingEntryIdentity(entries: ParsedSessionEntry[]): string {
  return entries.map((entry) => entry.id).join('\0');
}

function formatTurnContextText(turnContextText: TurnContextText): string {
  return `User:\n${turnContextText.userText}\n\nAssistant:\n${turnContextText.assistantText}`;
}

function addTurnContextPathDocument(
  documentsByContributingEntryIdentity: Map<string, PendingConversationDocument>,
  userEntry: ParsedSessionEntry,
  userText: string,
  path: PendingTurnContextPath,
): void {
  if (path.assistantTexts.length === 0) {
    return;
  }
  const contributingEntryIdentity = createContributingEntryIdentity(path.contributingEntries);
  if (documentsByContributingEntryIdentity.has(contributingEntryIdentity)) {
    return;
  }
  const turnContextText: TurnContextText = {
    userText,
    assistantText: path.assistantTexts.join('\n\n'),
  };
  documentsByContributingEntryIdentity.set(contributingEntryIdentity, {
    entry: userEntry,
    contributingEntries: path.contributingEntries,
    turnContextText,
    role: 'turn',
    documentKind: 'turn_context',
    summaryKind: null,
    evidenceKind: 'turn_context',
    evidencePart: 'content',
    isDenseSearchable: true,
    compactionFirstKeptEntryId: null,
    branchSummaryFromEntryId: null,
    toolCallId: null,
    toolName: null,
    toolCallEntryId: null,
    toolResultEntryId: null,
    toolError: null,
    preserveVerbatim: false,
    textRun: {
      text: formatTurnContextText(turnContextText),
      textRunIndex: 0,
      sourceBlockStart: null,
      sourceBlockEnd: null,
    },
  });
}

function splitTurnContextRoleText(
  text: string,
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): string[] {
  return splitConversationTextByTokens(text, tokenizer, maxTokens, 0).map((span) => span.content);
}

function readTurnContextPairSegment(
  segments: string[],
  pairIndex: number,
  pairCount: number,
): string {
  const segmentIndex = Math.min(
    segments.length - 1,
    Math.floor((pairIndex * segments.length) / pairCount),
  );
  const segment = segments[segmentIndex];
  if (!segment) {
    throw new Error(`Recall turn context segment missing at pair ${pairIndex}`);
  }
  return segment;
}

function createTurnContextTextsForBudgets(
  turnContextText: TurnContextText,
  tokenizer: ConversationTextTokenizer,
  userMaxTokens: number,
  assistantMaxTokens: number,
): string[] {
  const userSegments = splitTurnContextRoleText(turnContextText.userText, tokenizer, userMaxTokens);
  const assistantSegments = splitTurnContextRoleText(
    turnContextText.assistantText,
    tokenizer,
    assistantMaxTokens,
  );
  if (userSegments.length === 0 || assistantSegments.length === 0) {
    return [];
  }
  const pairCount = Math.max(userSegments.length, assistantSegments.length);
  return Array.from({ length: pairCount }, (_, pairIndex) =>
    formatTurnContextText({
      userText: readTurnContextPairSegment(userSegments, pairIndex, pairCount),
      assistantText: readTurnContextPairSegment(assistantSegments, pairIndex, pairCount),
    }),
  );
}

function throwTurnContextTokenBudgetError(maxTokens: number): never {
  throw new Error(
    `Recall turn context cannot fit both user and assistant text within maxTokens=${maxTokens}`,
  );
}

function createTokenBoundedTurnContextTexts(
  turnContextText: TurnContextText,
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): string[] {
  const completeTurnContext = formatTurnContextText(turnContextText);
  if (countConversationTokens(completeTurnContext, tokenizer) <= maxTokens) {
    return [completeTurnContext];
  }

  const framingTokenCount = countConversationTokens(
    formatTurnContextText({ userText: '', assistantText: '' }),
    tokenizer,
  );
  const availableTextTokens = maxTokens - framingTokenCount;
  if (availableTextTokens < 2) {
    throwTurnContextTokenBudgetError(maxTokens);
  }
  const userTokenCount = countConversationTokens(turnContextText.userText, tokenizer);
  const assistantTokenCount = countConversationTokens(turnContextText.assistantText, tokenizer);
  let userMaxTokens: number;
  let assistantMaxTokens: number;
  if (userTokenCount <= availableTextTokens - 1) {
    userMaxTokens = Math.max(1, userTokenCount);
    assistantMaxTokens = availableTextTokens - userMaxTokens;
  } else if (assistantTokenCount <= availableTextTokens - 1) {
    assistantMaxTokens = Math.max(1, assistantTokenCount);
    userMaxTokens = availableTextTokens - assistantMaxTokens;
  } else {
    userMaxTokens = Math.floor(availableTextTokens / 2);
    assistantMaxTokens = availableTextTokens - userMaxTokens;
  }

  while (userMaxTokens >= 1 && assistantMaxTokens >= 1) {
    const pairedTexts = createTurnContextTextsForBudgets(
      turnContextText,
      tokenizer,
      userMaxTokens,
      assistantMaxTokens,
    );
    if (pairedTexts.length === 0) {
      throwTurnContextTokenBudgetError(maxTokens);
    }
    if (pairedTexts.every((text) => countConversationTokens(text, tokenizer) <= maxTokens)) {
      return pairedTexts;
    }
    if (userMaxTokens >= assistantMaxTokens && userMaxTokens > 1) {
      userMaxTokens -= 1;
    } else if (assistantMaxTokens > 1) {
      assistantMaxTokens -= 1;
    } else if (userMaxTokens > 1) {
      userMaxTokens -= 1;
    } else {
      break;
    }
  }
  return throwTurnContextTokenBudgetError(maxTokens);
}

function createTokenBoundedTurnContextDocuments(
  pending: PendingConversationDocument,
  tokenizer: ConversationTextTokenizer,
  maxTokens: number,
): PendingConversationDocument[] {
  if (pending.documentKind !== 'turn_context') {
    return [pending];
  }
  return createTokenBoundedTurnContextTexts(pending.turnContextText, tokenizer, maxTokens).map(
    (text, textRunIndex) => ({
      ...pending,
      textRun: {
        ...pending.textRun,
        text,
        textRunIndex,
      },
    }),
  );
}

function createTurnContextDocuments(graph: ParsedSessionGraph): PendingConversationDocument[] {
  const documentsByContributingEntryIdentity = new Map<string, PendingConversationDocument>();
  for (const userEntry of graph.entries) {
    const userText = readTurnContextRoleText(userEntry, 'user');
    if (!userText) {
      continue;
    }
    const pendingPaths: PendingTurnContextPath[] = (
      graph.childEntryIdsById.get(userEntry.id) ?? []
    ).map((entryId) => ({
      entry: readSessionGraphEntry(graph, entryId),
      contributingEntries: [userEntry],
      assistantTexts: [],
    }));
    for (let pathIndex = 0; pathIndex < pendingPaths.length; pathIndex += 1) {
      const path = pendingPaths[pathIndex];
      if (!path) {
        continue;
      }
      const { entry } = path;
      const message = readSessionEntryMessage(entry);
      if (message?.role === 'user') {
        addTurnContextPathDocument(documentsByContributingEntryIdentity, userEntry, userText, path);
        continue;
      }

      const assistantText = readTurnContextRoleText(entry, 'assistant');
      const nextPath: PendingTurnContextPath = assistantText
        ? {
            entry,
            contributingEntries: [...path.contributingEntries, entry],
            assistantTexts: [...path.assistantTexts, assistantText],
          }
        : { ...path, entry };
      if (entry.id === graph.currentLeafId) {
        addTurnContextPathDocument(
          documentsByContributingEntryIdentity,
          userEntry,
          userText,
          nextPath,
        );
      }
      const childEntryIds = graph.childEntryIdsById.get(entry.id) ?? [];
      if (childEntryIds.length === 0) {
        addTurnContextPathDocument(
          documentsByContributingEntryIdentity,
          userEntry,
          userText,
          nextPath,
        );
      } else {
        pendingPaths.push(
          ...childEntryIds.map((entryId) => ({
            ...nextPath,
            entry: readSessionGraphEntry(graph, entryId),
          })),
        );
      }
    }
  }
  return Array.from(documentsByContributingEntryIdentity.values());
}

function createPendingConversationDocuments(
  graph: ParsedSessionGraph,
): PendingConversationDocument[] {
  const entryScopedDocuments = graph.entries.flatMap((entry) => {
    if (entry.type === 'message' && isUnknownRecord(entry.record.message)) {
      const message = entry.record.message;
      if (message.role === 'user' || message.role === 'assistant') {
        return [
          ...createConversationTextDocuments(entry, message.role, message.content),
          ...(message.role === 'assistant'
            ? createToolCallDocuments(graph, entry, message.content)
            : []),
        ];
      }
      if (message.role === 'toolResult') {
        return createToolResultDocuments(graph, entry, message);
      }
      if (message.role === 'bashExecution') {
        return createBashExecutionDocuments(entry, message);
      }
      return [];
    }
    if (entry.type === 'custom_message' && entry.record.display !== false) {
      return createConversationTextDocuments(entry, 'custom', entry.record.content);
    }
    if (entry.type === 'compaction') {
      return createSummaryTextDocument(entry, 'compaction');
    }
    if (entry.type === 'branch_summary') {
      return createSummaryTextDocument(entry, 'branch');
    }
    return [];
  });
  return [...entryScopedDocuments, ...createTurnContextDocuments(graph)];
}

function findContributingBranchPathLeafIds(
  graph: ParsedSessionGraph,
  contributingEntries: ParsedSessionEntry[],
): string[] {
  const firstEntry = contributingEntries[0];
  if (!firstEntry) {
    throw new Error('Recall document must have at least one contributing entry');
  }
  return (graph.branchPathLeafIdsByEntryId.get(firstEntry.id) ?? []).filter((leafId) =>
    contributingEntries
      .slice(1)
      .every((entry) => (graph.branchPathLeafIdsByEntryId.get(entry.id) ?? []).includes(leafId)),
  );
}

function findContributingCompactionEntryIds(
  graph: ParsedSessionGraph,
  contributingEntries: ParsedSessionEntry[],
): string[] {
  const compactionEntryIds = new Set<string>();
  for (const entry of contributingEntries) {
    for (const compactionEntryId of graph.compactedByEntryIdsByEntryId.get(entry.id) ?? []) {
      compactionEntryIds.add(compactionEntryId);
    }
  }
  return Array.from(compactionEntryIds);
}

function createSessionConversationChunks(
  context: SessionConversationChunkContext,
  pending: PendingConversationDocument,
): SessionConversationChunk[] {
  const { graph } = context;
  const entryId = createPiSessionEntryId(pending.entry.id);
  const contributingEntryIds = pending.contributingEntries.map((entry) =>
    createPiSessionEntryId(entry.id),
  );
  const contributingEntryIdentity = createContributingEntryIdentity(pending.contributingEntries);
  const branchPathLeafIds = findContributingBranchPathLeafIds(graph, pending.contributingEntries);
  const compactedByEntryIds = findContributingCompactionEntryIds(
    graph,
    pending.contributingEntries,
  );
  const sourceLineIndexes = pending.contributingEntries.map((entry) => entry.lineIndex);
  const textRunId = hashConversationValue(
    `${SESSION_CONVERSATION_SCHEMA_VERSION}\0${graph.header.id}\0${pending.entry.id}\0${contributingEntryIdentity}\0${pending.evidenceKind}\0${pending.evidencePart}\0${pending.textRun.textRunIndex}`,
  ).slice(0, 40);
  const chunkSpans = pending.preserveVerbatim
    ? splitVerbatimToolEvidenceByTokens(pending.textRun.text, context.tokenizer, context.maxTokens)
    : splitConversationTextByTokens(
        pending.textRun.text,
        context.tokenizer,
        context.maxTokens,
        context.overlapTokens,
      );
  const chunkIds = chunkSpans.map((span) =>
    hashConversationValue(
      `${SESSION_CONVERSATION_SCHEMA_VERSION}\0${textRunId}\0${span.characterStart}\0${span.characterEnd}`,
    ).slice(0, 40),
  );

  return chunkSpans.map((span, chunkIndex) => {
    const id = chunkIds[chunkIndex];
    if (!id) {
      throw new Error(`Recall chunk identity missing for ${textRunId}:${chunkIndex}`);
    }
    const previousSiblingId = chunkIds[chunkIndex - 1] ?? null;
    const nextSiblingId = chunkIds[chunkIndex + 1] ?? null;
    const siblingIds: string[] = [];
    if (previousSiblingId) {
      siblingIds.push(previousSiblingId);
    }
    if (nextSiblingId) {
      siblingIds.push(nextSiblingId);
    }
    return {
      schemaVersion: SESSION_CONVERSATION_SCHEMA_VERSION,
      documentKind: pending.documentKind,
      summaryKind: pending.summaryKind,
      evidenceKind: pending.evidenceKind,
      evidencePart: pending.evidencePart,
      isDenseSearchable: pending.isDenseSearchable,
      id,
      checksum: hashConversationValue(span.content),
      sessionId: context.sessionId,
      sessionPath: context.sessionPath,
      parentSessionPath: graph.header.parentSessionPath,
      cwd: graph.header.cwd,
      projectPath: graph.header.cwd,
      sessionName: graph.sessionName,
      entryId,
      parentEntryId: pending.entry.parentId ? createPiSessionEntryId(pending.entry.parentId) : null,
      childEntryIds: (graph.childEntryIdsById.get(pending.entry.id) ?? []).map(
        createPiSessionEntryId,
      ),
      contributingEntryIds,
      currentLeafId: context.currentLeafId,
      branchPathLeafIds: branchPathLeafIds.map(createPiSessionEntryId),
      isOnActiveBranch: pending.contributingEntries.every((entry) =>
        graph.activeBranchEntryIds.has(entry.id),
      ),
      isVisibleInActiveContext: pending.contributingEntries.every((entry) =>
        graph.activeContextEntryIds.has(entry.id),
      ),
      compactedByEntryIds: compactedByEntryIds.map(createPiSessionEntryId),
      compactionFirstKeptEntryId: pending.compactionFirstKeptEntryId
        ? createPiSessionEntryId(pending.compactionFirstKeptEntryId)
        : null,
      branchSummaryFromEntryId: pending.branchSummaryFromEntryId
        ? createPiSessionEntryId(pending.branchSummaryFromEntryId)
        : null,
      role: pending.role,
      timestamp: pending.entry.timestamp,
      sourceLineStart: Math.min(...sourceLineIndexes),
      sourceLineEnd: Math.max(...sourceLineIndexes),
      sourceBlockStart: pending.textRun.sourceBlockStart,
      sourceBlockEnd: pending.textRun.sourceBlockEnd,
      characterStart: span.characterStart,
      characterEnd: span.characterEnd,
      tokenStart: span.tokenStart,
      tokenEnd: span.tokenEnd,
      tokenCount: span.tokenCount,
      overlapTokenCount: span.overlapTokenCount,
      textRunId,
      textRunIndex: pending.textRun.textRunIndex,
      chunkIndex,
      chunkCount: chunkIds.length,
      siblingIds,
      previousSiblingId,
      nextSiblingId,
      toolCallId: pending.toolCallId,
      toolName: pending.toolName,
      toolCallEntryId: pending.toolCallEntryId
        ? createPiSessionEntryId(pending.toolCallEntryId)
        : null,
      toolResultEntryId: pending.toolResultEntryId
        ? createPiSessionEntryId(pending.toolResultEntryId)
        : null,
      toolError: pending.toolError,
      content: span.content,
    };
  });
}

/**
 * Reads exact-token atomic, turn-context, summary, and lexical-only tool evidence from one validated Pi session graph.
 * Source lines are one-based; block indexes are inclusive; character spans are half-open UTF-16
 * offsets; token spans are logical exact-token offsets whose sibling intersections equal overlap.
 */
export async function readSessionConversationChunks(
  sessionPath: string,
  options: SessionConversationChunkOptions,
): Promise<SessionConversationChunk[]> {
  const maxTokens = options.maxTokens ?? 1_024;
  const overlapTokens = options.overlapTokens ?? 128;
  if (
    !Number.isInteger(maxTokens) ||
    maxTokens < 1 ||
    maxTokens > 1_024 ||
    !Number.isInteger(overlapTokens) ||
    overlapTokens < 0 ||
    overlapTokens > 128 ||
    overlapTokens >= maxTokens
  ) {
    throw new Error(
      'Recall chunk policy invalid: maxTokens must be 1..1024 and overlapTokens must be 0..128 and smaller than maxTokens',
    );
  }

  const graph = await readValidatedSessionGraph(sessionPath);
  const context: SessionConversationChunkContext = {
    graph,
    sessionPath,
    tokenizer: options.tokenizer,
    maxTokens,
    overlapTokens,
    sessionId: { value: graph.header.id },
    currentLeafId: graph.currentLeafId ? createPiSessionEntryId(graph.currentLeafId) : null,
  };
  return createPendingConversationDocuments(graph)
    .flatMap((pending) =>
      createTokenBoundedTurnContextDocuments(pending, context.tokenizer, context.maxTokens),
    )
    .flatMap((pending) => createSessionConversationChunks(context, pending));
}
