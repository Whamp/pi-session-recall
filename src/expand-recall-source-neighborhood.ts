import { existsSync } from 'node:fs';
import { ZVecOpen, type ZVecCollection, type ZVecDoc } from '@zvec/zvec';

import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  readRecallGenerationManifest,
} from './recall-generation-manifest.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import {
  readRecallActiveGenerationPointer,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
import {
  createExpectedRecallPhysicalSourceManifest,
  parseRecallGenerationSearchDocument,
} from './recall-physical-source-generation.js';
import { readActiveTargetRecallManifestFingerprint } from './read-active-target-recall-generation.js';
import type { SessionConversationChunk } from './session-conversation-index.js';
import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';

/** Exact occurrence and bounded entry counts for one active source neighborhood. */
export interface ExpandRecallSourceNeighborhoodOptions {
  evidenceOccurrenceId: string;
  previousEntryCount?: number;
  nextEntryCount?: number;
  branchPathLeafEntryId?: string;
}

/** Exact indexed geometry retained for one chunk contributing to stitched evidence. */
export interface RecallSourceNeighborhoodEvidenceOccurrence {
  evidenceOccurrenceId: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  sourceBlockStart: number | null;
  sourceBlockEnd: number | null;
  characterStart: number;
  characterEnd: number;
  tokenStart: number;
  tokenEnd: number;
  textRunIndex: number;
  chunkIndex: number;
  chunkCount: number;
}

/** One source-faithful evidence part stitched without crossing provenance boundaries. */
export interface RecallSourceNeighborhoodEvidence {
  documentKind: SessionConversationChunk['documentKind'];
  summaryKind: SessionConversationChunk['summaryKind'];
  evidenceKind: SessionConversationChunk['evidenceKind'];
  evidencePart: SessionConversationChunk['evidencePart'];
  role: SessionConversationChunk['role'];
  content: string;
  contributingEntryIds: string[];
  branchPathLeafEntryIds: string[];
  currentLeafEntryId: string | null;
  compactedByEntryIds: string[];
  isOnActiveBranch: boolean;
  isVisibleInActiveContext: boolean;
  toolCallId: string | null;
  toolName: string | null;
  toolCallEntryId: string | null;
  toolResultEntryId: string | null;
  toolError: boolean | null;
  compactionFirstKeptEntryId: string | null;
  branchSummaryFromEntryId: string | null;
  occurrences: RecallSourceNeighborhoodEvidenceOccurrence[];
}

/** One counted graph entry, including a placeholder when no source evidence is returnable. */
export interface RecallSourceNeighborhoodEntry {
  entryAnchorId: string;
  entryId: string;
  parentEntryId: string | null;
  entryType: string;
  timestamp: string;
  sourceOrder: number;
  pathOrder: number;
  placeholder: boolean;
  evidence: RecallSourceNeighborhoodEvidence[];
}

/** Exact active-generation source neighborhood with requested and returned graph distances. */
export interface RecallSourceNeighborhood {
  anchorEvidenceOccurrenceId: string;
  physicalSourceIdentity: string;
  physicalSessionPath: string;
  sessionsRootRelativePath: string;
  logicalSessionOccurrenceId: string;
  rawSessionId: string;
  requestedEntryCounts: Readonly<{ previous: number; next: number }>;
  returnedEntryCounts: Readonly<{ previous: number; next: number }>;
  branchPathLeafEntryId: string | null;
  entries: RecallSourceNeighborhoodEntry[];
}

interface IndexedEntryAnchor {
  entryAnchorId: string;
  entryId: string;
  parentEntryId: string | null;
  childEntryIds: string[];
  branchPathLeafEntryIds: string[];
  evidenceOccurrenceIds: string[];
  entryType: string;
  timestamp: string;
  sourceOrder: number;
}

const LEXICAL_SOURCE_EXPANSION_FIELDS = [
  'recordKind',
  'physicalSourceIdentity',
  'sessionsRootRelativePath',
  'logicalSessionOccurrenceId',
  'rawSessionId',
  'entryAnchorId',
  'entryId',
  'parentEntryId',
  'childEntryIds',
  'branchPathLeafIds',
  'evidenceOccurrenceIds',
  'entryType',
  'timestamp',
  'sourceOrder',
  'recordJson',
] as const;

function readStringField(fields: Record<string, unknown>, fieldName: string): string {
  const value = fields[fieldName];
  if (typeof value === 'string') {
    return value;
  }
  throw new Error(`Recall source neighborhood incoherent ${fieldName}: expected a string`);
}

function readStringArrayField(fields: Record<string, unknown>, fieldName: string): string[] {
  const value = fields[fieldName];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'string')) {
    throw new Error(`Recall source neighborhood incoherent ${fieldName}: expected string values`);
  }
  return value;
}

function readIntegerField(fields: Record<string, unknown>, fieldName: string): number {
  const value = fields[fieldName];
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
    throw new Error(`Recall source neighborhood incoherent ${fieldName}: expected an integer`);
  }
  return value;
}

function parseIndexedEntryAnchor(fields: Record<string, unknown>): IndexedEntryAnchor {
  const parentEntryId = readStringField(fields, 'parentEntryId');
  return {
    entryAnchorId: readStringField(fields, 'entryAnchorId'),
    entryId: readStringField(fields, 'entryId'),
    parentEntryId: parentEntryId || null,
    childEntryIds: readStringArrayField(fields, 'childEntryIds'),
    branchPathLeafEntryIds: readStringArrayField(fields, 'branchPathLeafIds'),
    evidenceOccurrenceIds: readStringArrayField(fields, 'evidenceOccurrenceIds'),
    entryType: readStringField(fields, 'entryType'),
    timestamp: readStringField(fields, 'timestamp'),
    sourceOrder: readIntegerField(fields, 'sourceOrder'),
  };
}

function validateRequestedEntryCount(value: number | undefined, name: string): number {
  const count = value ?? 2;
  if (!Number.isSafeInteger(count) || count < 0 || count > 10) {
    throw new Error(`Recall source neighborhood ${name} invalid: expected an integer from 0 to 10`);
  }
  return count;
}

function validateExpansionOptions(
  options: Readonly<ExpandRecallSourceNeighborhoodOptions>,
): Readonly<{
  evidenceOccurrenceId: string;
  previousEntryCount: number;
  nextEntryCount: number;
  branchPathLeafEntryId: string | null;
}> {
  const evidenceOccurrenceId = options.evidenceOccurrenceId;
  if (!evidenceOccurrenceId.trim()) {
    throw new Error('Recall source neighborhood evidence occurrence ID must not be blank');
  }
  const branchPathLeafEntryId = options.branchPathLeafEntryId ?? null;
  if (branchPathLeafEntryId !== null && !branchPathLeafEntryId.trim()) {
    throw new Error('Recall source neighborhood branch-path leaf entry ID must not be blank');
  }
  return {
    evidenceOccurrenceId,
    previousEntryCount: validateRequestedEntryCount(
      options.previousEntryCount,
      'previous entry count',
    ),
    nextEntryCount: validateRequestedEntryCount(options.nextEntryCount, 'next entry count'),
    branchPathLeafEntryId,
  };
}

async function openActiveLexicalSourceStore(
  config: Readonly<RecallCoherentGenerationConfig>,
): Promise<
  Readonly<{
    generationId: string;
    collection: ZVecCollection;
  }>
> {
  const pointer = await readRecallActiveGenerationPointer(config.activeGenerationPointerPath);
  if (pointer === null) {
    throw new Error('Recall source neighborhood active generation pointer missing');
  }
  const generationId = pointer.activeGenerationId;
  const registryManifestFingerprint = await readActiveTargetRecallManifestFingerprint(
    config,
    generationId,
  );
  const generationDirectory = await resolveRecallGenerationDirectory(
    config.generationRootDirectory,
    generationId,
  );
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  if (existsSync(paths.recoveryRecordPath)) {
    throw new Error(`Recall source neighborhood recovery required for ${generationId}`);
  }
  const expectedManifest = createExpectedRecallPhysicalSourceManifest(config, generationId);
  const { manifest, fingerprint } = await readRecallGenerationManifest(paths.manifestPath);
  assertRecallGenerationManifestCompatible(manifest, expectedManifest, paths.manifestPath);
  const receipt = await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  if (receipt.generationId !== generationId || receipt.manifestFingerprint !== fingerprint) {
    throw new Error(
      `Recall source neighborhood validation receipt mismatch for active generation ${generationId}`,
    );
  }
  if (registryManifestFingerprint !== fingerprint) {
    throw new Error(
      `Recall source neighborhood registry manifest fingerprint mismatch for active generation ${generationId}`,
    );
  }
  return {
    generationId,
    collection: ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true }),
  };
}

function readExactEvidenceOccurrence(
  collection: ZVecCollection,
  evidenceOccurrenceId: string,
): Readonly<{ fields: Record<string, unknown>; chunk: SessionConversationChunk }> {
  const fetched = collection.fetchSync({
    ids: [evidenceOccurrenceId],
    outputFields: [...LEXICAL_SOURCE_EXPANSION_FIELDS],
    includeVector: false,
  });
  const document = fetched[evidenceOccurrenceId];
  if (document === undefined || document.fields.recordKind !== 'evidence') {
    throw new Error(
      `Recall source neighborhood evidence occurrence ID not found in active generation: ${evidenceOccurrenceId}`,
    );
  }
  return {
    fields: document.fields,
    chunk: parseRecallGenerationSearchDocument(evidenceOccurrenceId, document.fields.recordJson),
  };
}

function readLogicalSessionRecords(
  collection: ZVecCollection,
  logicalSessionOccurrenceId: string,
): Readonly<{
  anchorsByEntryId: Map<string, IndexedEntryAnchor>;
  evidenceByOccurrenceId: Map<string, SessionConversationChunk>;
}> {
  const documents: ZVecDoc[] = [];
  const logicalSessionFilter = `logicalSessionOccurrenceId = '${logicalSessionOccurrenceId}'`;
  visitExactZvecDocuments(
    collection,
    {
      filter: `(${logicalSessionFilter}) AND (recordKind = 'entry-anchor')`,
      uniquePartitionField: 'entryAnchorId',
      outputFields: LEXICAL_SOURCE_EXPANSION_FIELDS,
    },
    (document) => {
      documents.push(document);
    },
  );
  visitExactZvecDocuments(
    collection,
    {
      filter: `(${logicalSessionFilter}) AND (recordKind = 'evidence')`,
      uniquePartitionField: 'evidenceOccurrenceId',
      outputFields: LEXICAL_SOURCE_EXPANSION_FIELDS,
    },
    (document) => {
      documents.push(document);
    },
  );
  const anchorsByEntryId = new Map<string, IndexedEntryAnchor>();
  const evidenceByOccurrenceId = new Map<string, SessionConversationChunk>();
  for (const document of documents) {
    if (document.fields.recordKind === 'entry-anchor') {
      const anchor = parseIndexedEntryAnchor(document.fields);
      if (anchorsByEntryId.has(anchor.entryId)) {
        throw new Error(
          `Recall source neighborhood incoherent duplicate entry anchor: ${anchor.entryId}`,
        );
      }
      anchorsByEntryId.set(anchor.entryId, anchor);
    } else if (document.fields.recordKind === 'evidence') {
      evidenceByOccurrenceId.set(
        document.id,
        parseRecallGenerationSearchDocument(document.id, document.fields.recordJson),
      );
    } else {
      throw new Error(
        `Recall source neighborhood incoherent lexical/source record kind: ${String(document.fields.recordKind)}`,
      );
    }
  }
  return { anchorsByEntryId, evidenceByOccurrenceId };
}

function readRequiredEntryAnchor(
  anchorsByEntryId: ReadonlyMap<string, IndexedEntryAnchor>,
  entryId: string,
): IndexedEntryAnchor {
  const anchor = anchorsByEntryId.get(entryId);
  if (anchor === undefined) {
    throw new Error(`Recall source neighborhood incoherent entry anchor missing: ${entryId}`);
  }
  return anchor;
}

function collectPreviousEntryAnchors(
  anchor: IndexedEntryAnchor,
  anchorsByEntryId: ReadonlyMap<string, IndexedEntryAnchor>,
  count: number,
): IndexedEntryAnchor[] {
  const previous: IndexedEntryAnchor[] = [];
  let current = anchor;
  while (previous.length < count && current.parentEntryId !== null) {
    current = readRequiredEntryAnchor(anchorsByEntryId, current.parentEntryId);
    previous.push(current);
  }
  return previous.toReversed();
}

function selectDescendantEntryAnchor(
  current: IndexedEntryAnchor,
  anchorsByEntryId: ReadonlyMap<string, IndexedEntryAnchor>,
  branchPathLeafEntryId: string | null,
): IndexedEntryAnchor | null {
  if (current.childEntryIds.length === 0) {
    return null;
  }
  if (current.childEntryIds.length === 1) {
    return readRequiredEntryAnchor(anchorsByEntryId, current.childEntryIds[0] ?? '');
  }
  if (branchPathLeafEntryId === null) {
    throw new Error(
      `Recall source neighborhood forward path is ambiguous at ${current.entryId}; valid branch-path leaf entry IDs: ${current.branchPathLeafEntryIds.join(', ')}`,
    );
  }
  const matchingChildren = current.childEntryIds
    .map((entryId) => readRequiredEntryAnchor(anchorsByEntryId, entryId))
    .filter(
      (child) =>
        child.entryId === branchPathLeafEntryId ||
        child.branchPathLeafEntryIds.includes(branchPathLeafEntryId),
    );
  const selected = matchingChildren[0];
  if (matchingChildren.length !== 1 || selected === undefined) {
    throw new Error(
      `Recall source neighborhood branch-path leaf entry ID does not select a path from ${current.entryId}: ${branchPathLeafEntryId}`,
    );
  }
  return selected;
}

function collectNextEntryAnchors(
  anchor: IndexedEntryAnchor,
  anchorsByEntryId: ReadonlyMap<string, IndexedEntryAnchor>,
  count: number,
  branchPathLeafEntryId: string | null,
): IndexedEntryAnchor[] {
  if (
    branchPathLeafEntryId !== null &&
    anchor.entryId !== branchPathLeafEntryId &&
    !anchor.branchPathLeafEntryIds.includes(branchPathLeafEntryId)
  ) {
    throw new Error(
      `Recall source neighborhood branch-path leaf entry ID does not contain anchor ${anchor.entryId}: ${branchPathLeafEntryId}`,
    );
  }
  const next: IndexedEntryAnchor[] = [];
  let current = anchor;
  while (next.length < count) {
    const selected = selectDescendantEntryAnchor(current, anchorsByEntryId, branchPathLeafEntryId);
    if (selected === null) {
      break;
    }
    next.push(selected);
    current = selected;
  }
  return next;
}

function compareEvidenceSourceOrder(
  left: SessionConversationChunk,
  right: SessionConversationChunk,
): number {
  return (
    left.sourceLineStart - right.sourceLineStart ||
    (left.sourceBlockStart ?? -1) - (right.sourceBlockStart ?? -1) ||
    left.textRunIndex - right.textRunIndex ||
    left.characterStart - right.characterStart ||
    left.chunkIndex - right.chunkIndex ||
    left.id.localeCompare(right.id)
  );
}

function readStitchOverlapCharacters(
  earlier: SessionConversationChunk,
  later: SessionConversationChunk,
): number | null {
  if (
    earlier.entryId.value !== later.entryId.value ||
    earlier.documentKind !== later.documentKind ||
    earlier.summaryKind !== later.summaryKind ||
    earlier.evidenceKind !== later.evidenceKind ||
    earlier.evidencePart !== later.evidencePart ||
    earlier.role !== later.role ||
    earlier.textRunId !== later.textRunId ||
    earlier.textRunIndex !== later.textRunIndex ||
    earlier.nextSiblingId !== later.id ||
    later.previousSiblingId !== earlier.id ||
    later.chunkIndex !== earlier.chunkIndex + 1 ||
    later.tokenStart !== earlier.tokenEnd - later.overlapTokenCount
  ) {
    return null;
  }
  const overlapCharacters = earlier.characterEnd - later.characterStart;
  if (overlapCharacters < 0) {
    return null;
  }
  if (overlapCharacters === 0) {
    return later.overlapTokenCount === 0 ? 0 : null;
  }
  return overlapCharacters <= earlier.content.length &&
    overlapCharacters <= later.content.length &&
    later.overlapTokenCount > 0 &&
    earlier.content.slice(-overlapCharacters) === later.content.slice(0, overlapCharacters)
    ? overlapCharacters
    : null;
}

function createEvidenceOccurrence(
  chunk: SessionConversationChunk,
): RecallSourceNeighborhoodEvidenceOccurrence {
  return {
    evidenceOccurrenceId: chunk.id,
    sourceLineStart: chunk.sourceLineStart,
    sourceLineEnd: chunk.sourceLineEnd,
    sourceBlockStart: chunk.sourceBlockStart,
    sourceBlockEnd: chunk.sourceBlockEnd,
    characterStart: chunk.characterStart,
    characterEnd: chunk.characterEnd,
    tokenStart: chunk.tokenStart,
    tokenEnd: chunk.tokenEnd,
    textRunIndex: chunk.textRunIndex,
    chunkIndex: chunk.chunkIndex,
    chunkCount: chunk.chunkCount,
  };
}

function createStitchedEvidence(
  chunks: readonly SessionConversationChunk[],
): RecallSourceNeighborhoodEvidence {
  const first = chunks[0];
  if (first === undefined) {
    throw new Error('Recall source neighborhood cannot stitch an empty evidence part');
  }
  let content = first.content;
  for (let index = 1; index < chunks.length; index += 1) {
    const previous = chunks[index - 1];
    const current = chunks[index];
    if (previous === undefined || current === undefined) {
      throw new Error(`Recall source neighborhood chunk missing while stitching index ${index}`);
    }
    const overlapCharacters = readStitchOverlapCharacters(previous, current);
    if (overlapCharacters === null) {
      throw new Error(
        `Recall source neighborhood attempted unsafe stitching for ${previous.id} and ${current.id}`,
      );
    }
    content += current.content.slice(overlapCharacters);
  }
  return {
    documentKind: first.documentKind,
    summaryKind: first.summaryKind,
    evidenceKind: first.evidenceKind,
    evidencePart: first.evidencePart,
    role: first.role,
    content,
    contributingEntryIds: first.contributingEntryIds.map(({ value }) => value),
    branchPathLeafEntryIds: first.branchPathLeafIds.map(({ value }) => value),
    currentLeafEntryId: first.currentLeafId?.value ?? null,
    compactedByEntryIds: first.compactedByEntryIds.map(({ value }) => value),
    isOnActiveBranch: first.isOnActiveBranch,
    isVisibleInActiveContext: first.isVisibleInActiveContext,
    toolCallId: first.toolCallId,
    toolName: first.toolName,
    toolCallEntryId: first.toolCallEntryId?.value ?? null,
    toolResultEntryId: first.toolResultEntryId?.value ?? null,
    toolError: first.toolError,
    compactionFirstKeptEntryId: first.compactionFirstKeptEntryId?.value ?? null,
    branchSummaryFromEntryId: first.branchSummaryFromEntryId?.value ?? null,
    occurrences: chunks.map(createEvidenceOccurrence),
  };
}

function materializeEntryEvidence(
  anchor: IndexedEntryAnchor,
  evidenceByOccurrenceId: ReadonlyMap<string, SessionConversationChunk>,
): RecallSourceNeighborhoodEvidence[] {
  const sourceEvidence = anchor.evidenceOccurrenceIds
    .map((evidenceOccurrenceId) => {
      const chunk = evidenceByOccurrenceId.get(evidenceOccurrenceId);
      if (chunk === undefined) {
        throw new Error(
          `Recall source neighborhood incoherent named evidence occurrence missing: ${evidenceOccurrenceId}`,
        );
      }
      if (chunk.entryId.value !== anchor.entryId) {
        throw new Error(
          `Recall source neighborhood incoherent evidence entry mismatch for ${evidenceOccurrenceId}`,
        );
      }
      return chunk;
    })
    .filter(({ evidenceKind }) => evidenceKind !== 'turn_context')
    .toSorted(compareEvidenceSourceOrder);
  const groups: SessionConversationChunk[][] = [];
  for (const chunk of sourceEvidence) {
    const currentGroup = groups.at(-1);
    const previous = currentGroup?.at(-1);
    if (previous !== undefined && readStitchOverlapCharacters(previous, chunk) !== null) {
      currentGroup?.push(chunk);
    } else {
      groups.push([chunk]);
    }
  }
  return groups.map(createStitchedEvidence);
}

/** Expands one exact occurrence through active lexical/source entry anchors only. */
export async function expandRecallSourceNeighborhood(
  config: Readonly<RecallCoherentGenerationConfig>,
  options: Readonly<ExpandRecallSourceNeighborhoodOptions>,
): Promise<RecallSourceNeighborhood> {
  const request = validateExpansionOptions(options);
  const opened = await openActiveLexicalSourceStore(config);
  try {
    const anchorOccurrence = readExactEvidenceOccurrence(
      opened.collection,
      request.evidenceOccurrenceId,
    );
    const logicalSessionOccurrenceId = readStringField(
      anchorOccurrence.fields,
      'logicalSessionOccurrenceId',
    );
    const { anchorsByEntryId, evidenceByOccurrenceId } = readLogicalSessionRecords(
      opened.collection,
      logicalSessionOccurrenceId,
    );
    const anchor = readRequiredEntryAnchor(anchorsByEntryId, anchorOccurrence.chunk.entryId.value);
    const previous = collectPreviousEntryAnchors(
      anchor,
      anchorsByEntryId,
      request.previousEntryCount,
    );
    const next = collectNextEntryAnchors(
      anchor,
      anchorsByEntryId,
      request.nextEntryCount,
      request.branchPathLeafEntryId,
    );
    const selectedPath = [...previous, anchor, ...next];
    const entries = selectedPath.map((entry, pathOrder): RecallSourceNeighborhoodEntry => {
      const evidence = materializeEntryEvidence(entry, evidenceByOccurrenceId);
      return {
        entryAnchorId: entry.entryAnchorId,
        entryId: entry.entryId,
        parentEntryId: entry.parentEntryId,
        entryType: entry.entryType,
        timestamp: entry.timestamp,
        sourceOrder: entry.sourceOrder,
        pathOrder,
        placeholder: evidence.length === 0,
        evidence,
      };
    });
    return {
      anchorEvidenceOccurrenceId: request.evidenceOccurrenceId,
      physicalSourceIdentity: readStringField(anchorOccurrence.fields, 'physicalSourceIdentity'),
      physicalSessionPath: anchorOccurrence.chunk.sessionPath,
      sessionsRootRelativePath: readStringField(
        anchorOccurrence.fields,
        'sessionsRootRelativePath',
      ),
      logicalSessionOccurrenceId,
      rawSessionId: readStringField(anchorOccurrence.fields, 'rawSessionId'),
      requestedEntryCounts: {
        previous: request.previousEntryCount,
        next: request.nextEntryCount,
      },
      returnedEntryCounts: { previous: previous.length, next: next.length },
      branchPathLeafEntryId: request.branchPathLeafEntryId,
      entries,
    };
  } finally {
    opened.collection.closeSync();
  }
}
