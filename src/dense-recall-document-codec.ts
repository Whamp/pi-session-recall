import { createHash } from 'node:crypto';

import { ZVecDataType, type ZVecDoc, type ZVecFieldSchema } from '@zvec/zvec';

import { RecallProjectIdentitySource } from './enums.js';
import {
  isCanonicalRepositoryIdentity,
  parseProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import type {
  PiSessionEntryId,
  PiSessionId,
  SessionConversationChunk,
} from './session-conversation-index.js';

/** Scalar schema for dense recall documents; no field has a full-text index. */
export const DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS: readonly ZVecFieldSchema[] = Object.freeze([
  { name: 'schemaVersion', dataType: ZVecDataType.INT32 },
  { name: 'documentKind', dataType: ZVecDataType.STRING },
  { name: 'summaryKind', dataType: ZVecDataType.STRING },
  { name: 'evidenceKind', dataType: ZVecDataType.STRING },
  { name: 'evidencePart', dataType: ZVecDataType.STRING },
  { name: 'checksum', dataType: ZVecDataType.STRING },
  { name: 'sessionId', dataType: ZVecDataType.STRING },
  { name: 'sessionPath', dataType: ZVecDataType.STRING },
  { name: 'parentSessionPath', dataType: ZVecDataType.STRING },
  { name: 'cwd', dataType: ZVecDataType.STRING },
  { name: 'projectPath', dataType: ZVecDataType.STRING },
  { name: 'projectIdentity', dataType: ZVecDataType.STRING },
  { name: 'projectIdentityDigest', dataType: ZVecDataType.STRING },
  { name: 'projectIdentitySource', dataType: ZVecDataType.STRING },
  { name: 'sessionName', dataType: ZVecDataType.STRING },
  { name: 'entryId', dataType: ZVecDataType.STRING },
  { name: 'parentEntryId', dataType: ZVecDataType.STRING },
  { name: 'childEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'contributingEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'currentLeafId', dataType: ZVecDataType.STRING },
  { name: 'branchPathLeafIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'isOnActiveBranch', dataType: ZVecDataType.BOOL },
  { name: 'isVisibleInActiveContext', dataType: ZVecDataType.BOOL },
  { name: 'compactedByEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'compactionFirstKeptEntryId', dataType: ZVecDataType.STRING },
  { name: 'branchSummaryFromEntryId', dataType: ZVecDataType.STRING },
  { name: 'role', dataType: ZVecDataType.STRING },
  { name: 'timestamp', dataType: ZVecDataType.STRING },
  { name: 'sourceLineStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceLineEnd', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockEnd', dataType: ZVecDataType.INT32 },
  { name: 'characterStart', dataType: ZVecDataType.INT32 },
  { name: 'characterEnd', dataType: ZVecDataType.INT32 },
  { name: 'tokenStart', dataType: ZVecDataType.INT32 },
  { name: 'tokenEnd', dataType: ZVecDataType.INT32 },
  { name: 'tokenCount', dataType: ZVecDataType.INT32 },
  { name: 'overlapTokenCount', dataType: ZVecDataType.INT32 },
  { name: 'textRunId', dataType: ZVecDataType.STRING },
  { name: 'textRunIndex', dataType: ZVecDataType.INT32 },
  { name: 'chunkIndex', dataType: ZVecDataType.INT32 },
  { name: 'chunkCount', dataType: ZVecDataType.INT32 },
  { name: 'siblingIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'previousSiblingId', dataType: ZVecDataType.STRING },
  { name: 'nextSiblingId', dataType: ZVecDataType.STRING },
  { name: 'content', dataType: ZVecDataType.STRING },
]);

function serializeNullableEntryId(value: PiSessionEntryId | null): string {
  return value?.value ?? '';
}

/** Serializes one dense recall document without tool or full-text fields. */
export function serializeDenseRecallDocumentFields(
  document: SessionConversationChunk,
): Record<string, unknown> {
  return {
    schemaVersion: document.schemaVersion,
    documentKind: document.documentKind,
    summaryKind: document.summaryKind ?? '',
    evidenceKind: document.evidenceKind,
    evidencePart: document.evidencePart,
    checksum: document.checksum,
    sessionId: document.sessionId.value,
    sessionPath: document.sessionPath,
    parentSessionPath: document.parentSessionPath ?? '',
    cwd: document.cwd,
    projectPath: document.projectPath,
    projectIdentity: document.projectAttribution?.projectIdentity ?? '',
    projectIdentityDigest: document.projectAttribution
      ? createHash('sha256').update(document.projectAttribution.projectIdentity).digest('hex')
      : '',
    projectIdentitySource: document.projectAttribution?.identitySource ?? '',
    sessionName: document.sessionName,
    entryId: document.entryId.value,
    parentEntryId: serializeNullableEntryId(document.parentEntryId),
    childEntryIds: document.childEntryIds.map((id) => id.value),
    contributingEntryIds: document.contributingEntryIds.map((id) => id.value),
    currentLeafId: serializeNullableEntryId(document.currentLeafId),
    branchPathLeafIds: document.branchPathLeafIds.map((id) => id.value),
    isOnActiveBranch: document.isOnActiveBranch,
    isVisibleInActiveContext: document.isVisibleInActiveContext,
    compactedByEntryIds: document.compactedByEntryIds.map((id) => id.value),
    compactionFirstKeptEntryId: serializeNullableEntryId(document.compactionFirstKeptEntryId),
    branchSummaryFromEntryId: serializeNullableEntryId(document.branchSummaryFromEntryId),
    role: document.role,
    timestamp: document.timestamp,
    sourceLineStart: document.sourceLineStart,
    sourceLineEnd: document.sourceLineEnd,
    sourceBlockStart: document.sourceBlockStart ?? -1,
    sourceBlockEnd: document.sourceBlockEnd ?? -1,
    characterStart: document.characterStart,
    characterEnd: document.characterEnd,
    tokenStart: document.tokenStart,
    tokenEnd: document.tokenEnd,
    tokenCount: document.tokenCount,
    overlapTokenCount: document.overlapTokenCount,
    textRunId: document.textRunId,
    textRunIndex: document.textRunIndex,
    chunkIndex: document.chunkIndex,
    chunkCount: document.chunkCount,
    siblingIds: document.siblingIds,
    previousSiblingId: document.previousSiblingId ?? '',
    nextSiblingId: document.nextSiblingId ?? '',
    content: document.content,
  };
}

function readStringField(fields: Record<string, unknown>, name: string): string {
  const value = fields[name];
  if (typeof value !== 'string') {
    throw new Error(`Dense recall field invalid for ${name}: expected string`);
  }
  return value;
}

function readNumberField(fields: Record<string, unknown>, name: string): number {
  const value = fields[name];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Dense recall field invalid for ${name}: expected finite number`);
  }
  return value;
}

function readBooleanField(fields: Record<string, unknown>, name: string): boolean {
  const value = fields[name];
  if (typeof value !== 'boolean') {
    throw new Error(`Dense recall field invalid for ${name}: expected boolean`);
  }
  return value;
}

function readStringArrayField(fields: Record<string, unknown>, name: string): string[] {
  const value = fields[name];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'string')) {
    throw new Error(`Dense recall field invalid for ${name}: expected string array`);
  }
  return value.map(String);
}

function parseNullableString(value: string): string | null {
  return value || null;
}

function parseNullableEntryId(value: string): PiSessionEntryId | null {
  return value ? { value } : null;
}

function parseDenseRecallProjectIdentitySource(value: string): RecallProjectIdentitySource {
  switch (value) {
    case 'git_origin':
      return RecallProjectIdentitySource.GIT_ORIGIN;
    case 'git_common_directory':
      return RecallProjectIdentitySource.GIT_COMMON_DIRECTORY;
    case 'non_git_session_origin':
      return RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN;
    case 'configured_project_lineage':
      return RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE;
    default:
      throw new Error(`Dense recall project identity source invalid: ${value}`);
  }
}

function parseDenseRecallDocumentKind(value: string): SessionConversationChunk['documentKind'] {
  if (value === 'conversation' || value === 'turn_context' || value === 'summary') {
    return value;
  }
  throw new Error(`Dense recall document kind invalid: ${value}`);
}

function parseDenseRecallSummaryKind(value: string): SessionConversationChunk['summaryKind'] {
  if (value === '') {
    return null;
  }
  if (value === 'compaction' || value === 'branch') {
    return value;
  }
  throw new Error(`Dense recall summary kind invalid: ${value}`);
}

function parseDenseRecallEvidenceKind(value: string): SessionConversationChunk['evidenceKind'] {
  if (
    value === 'conversation' ||
    value === 'turn_context' ||
    value === 'compaction_summary' ||
    value === 'branch_summary'
  ) {
    return value;
  }
  throw new Error(`Dense recall evidence kind invalid: ${value}`);
}

function parseDenseRecallRole(value: string): SessionConversationChunk['role'] {
  if (
    value === 'user' ||
    value === 'assistant' ||
    value === 'turn' ||
    value === 'summary' ||
    value === 'custom'
  ) {
    return value;
  }
  throw new Error(`Dense recall role invalid: ${value}`);
}

function parseProjectAttribution(
  projectIdentity: string,
  identitySource: string,
): ResolvedProjectIdentity | null {
  if (!projectIdentity && !identitySource) {
    return null;
  }
  if (!projectIdentity || !identitySource) {
    throw new Error('Dense recall project attribution invalid: identity and source must coexist');
  }
  const parsedIdentity = parseProjectIdentity(projectIdentity);
  const parsedSource = parseDenseRecallProjectIdentitySource(identitySource);
  if (parsedSource === RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN) {
    return { projectIdentity: parsedIdentity, identitySource: parsedSource };
  }
  if (!isCanonicalRepositoryIdentity(parsedIdentity)) {
    throw new Error(
      `Dense recall project attribution invalid: ${parsedSource} requires repository identity`,
    );
  }
  return { projectIdentity: parsedIdentity, identitySource: parsedSource };
}

/** Deserializes one dense recall document with tool provenance fixed to absent. */
export function deserializeDenseRecallDocumentFields(document: ZVecDoc): SessionConversationChunk {
  const fields: Record<string, unknown> = document.fields;
  return {
    schemaVersion: readNumberField(fields, 'schemaVersion'),
    documentKind: parseDenseRecallDocumentKind(readStringField(fields, 'documentKind')),
    summaryKind: parseDenseRecallSummaryKind(readStringField(fields, 'summaryKind')),
    evidenceKind: parseDenseRecallEvidenceKind(readStringField(fields, 'evidenceKind')),
    evidencePart: 'content',
    id: document.id,
    checksum: readStringField(fields, 'checksum'),
    sessionId: { value: readStringField(fields, 'sessionId') } satisfies PiSessionId,
    sessionPath: readStringField(fields, 'sessionPath'),
    parentSessionPath: parseNullableString(readStringField(fields, 'parentSessionPath')),
    cwd: readStringField(fields, 'cwd'),
    projectPath: readStringField(fields, 'projectPath'),
    projectAttribution: parseProjectAttribution(
      readStringField(fields, 'projectIdentity'),
      readStringField(fields, 'projectIdentitySource'),
    ),
    sessionName: readStringField(fields, 'sessionName'),
    entryId: { value: readStringField(fields, 'entryId') } satisfies PiSessionEntryId,
    parentEntryId: parseNullableEntryId(readStringField(fields, 'parentEntryId')),
    childEntryIds: readStringArrayField(fields, 'childEntryIds').map((value) => ({ value })),
    contributingEntryIds: readStringArrayField(fields, 'contributingEntryIds').map((value) => ({
      value,
    })),
    currentLeafId: parseNullableEntryId(readStringField(fields, 'currentLeafId')),
    branchPathLeafIds: readStringArrayField(fields, 'branchPathLeafIds').map((value) => ({
      value,
    })),
    isOnActiveBranch: readBooleanField(fields, 'isOnActiveBranch'),
    isVisibleInActiveContext: readBooleanField(fields, 'isVisibleInActiveContext'),
    compactedByEntryIds: readStringArrayField(fields, 'compactedByEntryIds').map((value) => ({
      value,
    })),
    compactionFirstKeptEntryId: parseNullableEntryId(
      readStringField(fields, 'compactionFirstKeptEntryId'),
    ),
    branchSummaryFromEntryId: parseNullableEntryId(
      readStringField(fields, 'branchSummaryFromEntryId'),
    ),
    role: parseDenseRecallRole(readStringField(fields, 'role')),
    timestamp: readStringField(fields, 'timestamp'),
    sourceLineStart: readNumberField(fields, 'sourceLineStart'),
    sourceLineEnd: readNumberField(fields, 'sourceLineEnd'),
    sourceBlockStart: parseNullableSourceBlock(readNumberField(fields, 'sourceBlockStart')),
    sourceBlockEnd: parseNullableSourceBlock(readNumberField(fields, 'sourceBlockEnd')),
    characterStart: readNumberField(fields, 'characterStart'),
    characterEnd: readNumberField(fields, 'characterEnd'),
    tokenStart: readNumberField(fields, 'tokenStart'),
    tokenEnd: readNumberField(fields, 'tokenEnd'),
    tokenCount: readNumberField(fields, 'tokenCount'),
    overlapTokenCount: readNumberField(fields, 'overlapTokenCount'),
    textRunId: readStringField(fields, 'textRunId'),
    textRunIndex: readNumberField(fields, 'textRunIndex'),
    chunkIndex: readNumberField(fields, 'chunkIndex'),
    chunkCount: readNumberField(fields, 'chunkCount'),
    siblingIds: readStringArrayField(fields, 'siblingIds'),
    previousSiblingId: parseNullableString(readStringField(fields, 'previousSiblingId')),
    nextSiblingId: parseNullableString(readStringField(fields, 'nextSiblingId')),
    content: readStringField(fields, 'content'),
  };
}

function parseNullableSourceBlock(value: number): number | null {
  return value < 0 ? null : value;
}
