import { createHash } from 'node:crypto';
import { isAbsolute, relative, resolve, sep } from 'node:path';

/** Stable physical source identity and normalized path relative to the configured sessions root. */
export interface RecallPhysicalSourceIdentity {
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
}

/** Complete source position that identifies one immutable entry anchor. */
export interface RecallEntryAnchorIdentityInput {
  physicalSourceIdentity: string;
  logicalSessionOccurrenceId: string;
  entryId: string;
  sourceLine: number;
  startByte: number;
  endByte: number;
}

/** Complete source and chunk geometry that identifies one evidence occurrence. */
export interface RecallEvidenceOccurrenceIdentityInput {
  physicalSourceIdentity: string;
  logicalSessionOccurrenceId: string;
  entryId: string;
  evidencePart: string;
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
}

function createRecallSourceIdentityDigest(domain: string, parts: readonly unknown[]): string {
  return createHash('sha256')
    .update(JSON.stringify([domain, ...parts]))
    .digest('base64url');
}

function assertNonemptyRecallSourceIdentity(value: string, fieldName: string): void {
  if (!value) {
    throw new Error(`Recall source identity ${fieldName} invalid: expected a nonempty string`);
  }
}

function assertNonnegativeSafeInteger(value: number, fieldName: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Recall source identity ${fieldName} invalid: expected a nonnegative integer`);
  }
}

/**
 * Derives physical source identity only from the normalized sessions-root-relative path.
 * Moving the sessions root preserves identity; changing a path inside the root changes identity.
 */
export function resolveRecallPhysicalSourceIdentity(
  sessionsRootDirectory: string,
  physicalSessionPath: string,
): RecallPhysicalSourceIdentity {
  const normalizedRoot = resolve(sessionsRootDirectory);
  const normalizedSource = resolve(physicalSessionPath);
  const relativePath = relative(normalizedRoot, normalizedSource);
  if (
    !relativePath ||
    relativePath === '..' ||
    relativePath.startsWith(`..${sep}`) ||
    isAbsolute(relativePath)
  ) {
    throw new Error(
      `Recall physical source path escapes configured sessions root: ${physicalSessionPath}`,
    );
  }
  const sessionsRootRelativePath = relativePath.split(sep).join('/');
  return {
    physicalSourceIdentity: `source_${createRecallSourceIdentityDigest(
      'physical_source_identity_v1',
      [sessionsRootRelativePath],
    )}`,
    sessionsRootRelativePath,
  };
}

/** Identifies one logical session occurrence by physical source and complete-header source line. */
export function createRecallLogicalSessionOccurrenceId(
  physicalSourceIdentity: string,
  headerSourceLine: number,
): string {
  assertNonemptyRecallSourceIdentity(physicalSourceIdentity, 'physical source');
  if (!Number.isSafeInteger(headerSourceLine) || headerSourceLine < 1) {
    throw new Error(
      'Recall source identity header source line invalid: expected a positive integer',
    );
  }
  return `logical_${createRecallSourceIdentityDigest('logical_session_occurrence_v1', [
    physicalSourceIdentity,
    headerSourceLine,
  ])}`;
}

/** Identifies one immutable entry anchor by source location rather than entry text. */
export function createRecallEntryAnchorId(input: RecallEntryAnchorIdentityInput): string {
  assertNonemptyRecallSourceIdentity(input.physicalSourceIdentity, 'physical source');
  assertNonemptyRecallSourceIdentity(input.logicalSessionOccurrenceId, 'logical occurrence');
  assertNonemptyRecallSourceIdentity(input.entryId, 'entry ID');
  assertNonnegativeSafeInteger(input.sourceLine, 'entry source line');
  assertNonnegativeSafeInteger(input.startByte, 'entry start byte');
  assertNonnegativeSafeInteger(input.endByte, 'entry end byte');
  if (input.sourceLine < 1 || input.endByte <= input.startByte) {
    throw new Error(
      'Recall source identity entry geometry invalid: expected a nonempty source span',
    );
  }
  return `anchor_${createRecallSourceIdentityDigest('entry_anchor_v1', [
    input.physicalSourceIdentity,
    input.logicalSessionOccurrenceId,
    input.entryId,
    input.sourceLine,
    input.startByte,
    input.endByte,
  ])}`;
}

/** Identifies one evidence occurrence by source and chunk geometry rather than content. */
export function createRecallEvidenceOccurrenceId(
  input: RecallEvidenceOccurrenceIdentityInput,
): string {
  assertNonemptyRecallSourceIdentity(input.physicalSourceIdentity, 'physical source');
  assertNonemptyRecallSourceIdentity(input.logicalSessionOccurrenceId, 'logical occurrence');
  assertNonemptyRecallSourceIdentity(input.entryId, 'entry ID');
  assertNonemptyRecallSourceIdentity(input.evidencePart, 'evidence part');
  for (const [fieldName, value] of Object.entries({
    sourceLineStart: input.sourceLineStart,
    sourceLineEnd: input.sourceLineEnd,
    characterStart: input.characterStart,
    characterEnd: input.characterEnd,
    tokenStart: input.tokenStart,
    tokenEnd: input.tokenEnd,
    textRunIndex: input.textRunIndex,
    chunkIndex: input.chunkIndex,
  })) {
    assertNonnegativeSafeInteger(value, fieldName);
  }
  if (input.sourceLineStart < 1 || input.sourceLineEnd < input.sourceLineStart) {
    throw new Error('Recall source identity evidence source lines invalid');
  }
  if (input.characterEnd <= input.characterStart || input.tokenEnd < input.tokenStart) {
    throw new Error('Recall source identity evidence geometry invalid');
  }
  return `occurrence_${createRecallSourceIdentityDigest('evidence_occurrence_v1', [
    input.physicalSourceIdentity,
    input.logicalSessionOccurrenceId,
    input.entryId,
    input.evidencePart,
    input.sourceLineStart,
    input.sourceLineEnd,
    input.sourceBlockStart,
    input.sourceBlockEnd,
    input.characterStart,
    input.characterEnd,
    input.tokenStart,
    input.tokenEnd,
    input.textRunIndex,
    input.chunkIndex,
  ])}`;
}
