import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecIndexType, ZVecOpen, type ZVecCollection, type ZVecStatus } from '@zvec/zvec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallSessionProjectionKind } from './enums.js';
import { createRecallSessionProjectionBaselineFromImport } from './create-recall-session-projection-baseline.js';
import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  createRecallGenerationManifest,
  readRecallGenerationManifest,
} from './recall-generation-manifest.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import {
  createRecallEntryAnchorId,
  createRecallEvidenceOccurrenceId,
  createRecallLogicalSessionOccurrenceId,
  resolveRecallPhysicalSourceIdentity,
} from './recall-source-identity.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { LogicalSessionProjection } from './recall-session-projection.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
  type SessionConversationLogicalSession,
} from './session-conversation-index.js';

/** Fixed physical-source snapshot selected for one resumable inactive target generation. */
export interface CreateRecallGenerationFromPhysicalSourcesOptions {
  generationId: string;
  physicalSessionPaths: readonly string[];
  validatedVectorSourceGenerationId?: string;
  signal?: AbortSignal;
}

/** Source-faithful lexical evidence returned from one explicitly named inactive generation. */
export interface RecallGenerationLexicalEvidence {
  evidenceOccurrenceId: string;
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  logicalSessionOccurrenceId: string;
  rawSessionId: string;
  entryAnchorId: string;
  entryId: string;
  parentEntryId: string | null;
  evidenceKind: SessionConversationChunk['evidenceKind'];
  evidencePart: SessionConversationChunk['evidencePart'];
  isDenseSearchable: boolean;
  projectIdentity: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  content: string;
}

const recallGenerationLexicalEvidenceSchema = Type.Object(
  {
    evidenceOccurrenceId: Type.String({ minLength: 1 }),
    physicalSourceIdentity: Type.String({ minLength: 1 }),
    sessionsRootRelativePath: Type.String({ minLength: 1 }),
    logicalSessionOccurrenceId: Type.String({ minLength: 1 }),
    rawSessionId: Type.String({ minLength: 1 }),
    entryAnchorId: Type.String({ minLength: 1 }),
    entryId: Type.String({ minLength: 1 }),
    parentEntryId: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    evidenceKind: Type.Union([
      Type.Literal('conversation'),
      Type.Literal('turn_context'),
      Type.Literal('compaction_summary'),
      Type.Literal('branch_summary'),
      Type.Literal('tool_call'),
      Type.Literal('tool_result'),
      Type.Literal('bash_execution'),
    ]),
    evidencePart: Type.Union([
      Type.Literal('content'),
      Type.Literal('name'),
      Type.Literal('arguments'),
      Type.Literal('result'),
      Type.Literal('command'),
      Type.Literal('output'),
    ]),
    isDenseSearchable: Type.Boolean(),
    projectIdentity: Type.String(),
    sourceLineStart: Type.Integer({ minimum: 1 }),
    sourceLineEnd: Type.Integer({ minimum: 1 }),
    content: Type.String(),
  },
  { additionalProperties: false },
);

/** Runtime dependencies needed to materialize and fault-probe a fixed source snapshot. */
export interface RecallPhysicalSourceGenerationDependencies {
  tokenizer: ConversationTextTokenizer;
  embeddingProvider: RecallEmbeddingProvider;
  resolveProjectIdentity(workingDirectory: string): Promise<ResolvedProjectIdentity | null>;
  fixedSnapshotBuildFault?: (
    stage: 'after-snapshot-capture' | 'after-dense-write' | 'before-validation-receipt',
    context: Readonly<{ generationDirectory: string; physicalSourceIdentity?: string }>,
  ) => void | Promise<void>;
}

/** One dense row contract captured before its vector is resolved. */
export interface RecallGenerationDenseExpectation {
  id: string;
  fields: Record<string, unknown>;
  embeddingInput: string;
}

/** Exact target rows captured from one canonical physical-source import. */
export interface MaterializedRecallPhysicalSourceGeneration {
  lexicalSource: Array<{ id: string; fields: Record<string, unknown> }>;
  dense: RecallGenerationDenseExpectation[];
  logicalSessionProjections: Array<{ id: string; fields: Record<string, unknown> }>;
  physicalSessionProjection: { id: string; fields: Record<string, unknown> };
  physicalSourceIdentities: string[];
  logicalSessionOccurrenceIds: string[];
}

/** Creates the configured fixed manifest shared by target build and read operations. */
export function createExpectedRecallPhysicalSourceManifest(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
) {
  return createRecallGenerationManifest({
    generationId,
    embeddingProfileId: config.embeddingProfileId,
    embeddingProfile: config.embeddingProfile,
    projectLineages: config.projectLineages,
    ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
  });
}

function createRecallPhysicalSourceMembership(recordIds: readonly string[]): Readonly<{
  count: number;
  digest: string;
}> {
  const sortedRecordIds = [...recordIds].toSorted();
  return {
    count: sortedRecordIds.length,
    digest: createHash('sha256').update(JSON.stringify(sortedRecordIds)).digest('hex'),
  };
}

function assertCheckedZvecStatuses(
  operation: string,
  recordIds: readonly string[],
  statuses: readonly ZVecStatus[],
): void {
  if (statuses.length !== recordIds.length) {
    throw new Error(
      `Recall physical source generation ${operation} status mismatch: expected ${recordIds.length}, received ${statuses.length}`,
    );
  }
  for (const [index, status] of statuses.entries()) {
    if (!status.ok) {
      throw new Error(
        `Recall physical source generation ${operation} failed for ${recordIds[index] ?? 'unknown record'}: ${status.message}`,
      );
    }
  }
}

function findLogicalSessionForChunk(
  logicalSessions: readonly SessionConversationLogicalSession[],
  chunk: SessionConversationChunk,
): SessionConversationLogicalSession {
  const matches = logicalSessions.filter(
    ({ sourceLineStart, sourceLineEnd }) =>
      chunk.sourceLineStart >= sourceLineStart && chunk.sourceLineStart <= sourceLineEnd,
  );
  const logicalSession = matches[0];
  if (matches.length !== 1 || logicalSession === undefined) {
    throw new Error(
      `Recall physical source generation logical occurrence ambiguous at ${chunk.sessionPath}:${chunk.sourceLineStart}`,
    );
  }
  return logicalSession;
}

function findLogicalProjection(
  projections: readonly LogicalSessionProjection[],
  logicalSession: SessionConversationLogicalSession,
): LogicalSessionProjection {
  const projection = projections.find(
    ({ rawSessionId, headerDescriptor }) =>
      rawSessionId === logicalSession.sessionId &&
      headerDescriptor.sourceLine === logicalSession.sourceLineStart,
  );
  if (projection === undefined) {
    throw new Error(
      `Recall physical source generation projection missing for ${logicalSession.sessionId}@${logicalSession.sourceLineStart}`,
    );
  }
  return projection;
}

function createBranchLeafIdsByEntryId(
  logicalSession: SessionConversationLogicalSession,
): Map<string, string[]> {
  const childIdsByEntryId = new Map<string, string[]>();
  for (const [index, entryId] of logicalSession.entryIds.entries()) {
    const parentEntryId = logicalSession.parentEntryIds[index] ?? null;
    if (parentEntryId !== null) {
      const childIds = childIdsByEntryId.get(parentEntryId) ?? [];
      childIds.push(entryId);
      childIdsByEntryId.set(parentEntryId, childIds);
    }
  }
  const leafIds = logicalSession.entryIds.filter(
    (entryId) => (childIdsByEntryId.get(entryId) ?? []).length === 0,
  );
  const result = new Map<string, string[]>();
  for (const entryId of logicalSession.entryIds) {
    const descendants = leafIds.filter((leafId) => {
      let current: string | null = leafId;
      while (current !== null) {
        if (current === entryId) {
          return true;
        }
        const currentIndex = logicalSession.entryIds.indexOf(current);
        current = currentIndex < 0 ? null : (logicalSession.parentEntryIds[currentIndex] ?? null);
      }
      return false;
    });
    result.set(entryId, descendants.toSorted());
  }
  return result;
}

function createCommonLexicalFields(options: {
  generationId: string;
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  logicalSessionOccurrenceId: string;
  rawSessionId: string;
  headerSourceLine: number;
  entryAnchorId: string;
  entryId: string;
  parentEntryId: string | null;
  childEntryIds: readonly string[];
  branchPathLeafIds: readonly string[];
  sourceOrder: number;
  entryType: string;
  timestamp: string;
  entryStartByte: number;
  entryEndByte: number;
}): Record<string, unknown> {
  return {
    schemaVersion: 1,
    generationId: options.generationId,
    physicalSourceIdentity: options.physicalSourceIdentity,
    sessionsRootRelativePath: options.sessionsRootRelativePath,
    logicalSessionOccurrenceId: options.logicalSessionOccurrenceId,
    rawSessionId: options.rawSessionId,
    headerSourceLine: options.headerSourceLine,
    entryAnchorId: options.entryAnchorId,
    entryId: options.entryId,
    parentEntryId: options.parentEntryId ?? '',
    childEntryIds: [...options.childEntryIds],
    branchPathLeafIds: [...options.branchPathLeafIds],
    sourceOrder: options.sourceOrder,
    entryType: options.entryType,
    timestamp: options.timestamp,
    entryStartByte: options.entryStartByte,
    entryEndByte: options.entryEndByte,
  };
}

/** Materializes expected target rows through one canonical import and tokenization pass. */
export async function materializeRecallPhysicalSourceGeneration(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  physicalSessionPath: string,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  sourceReadPath = physicalSessionPath,
): Promise<MaterializedRecallPhysicalSourceGeneration> {
  const physicalSource = resolveRecallPhysicalSourceIdentity(
    config.sessionsDirectory,
    physicalSessionPath,
  );
  const sourceByteSize = (await stat(sourceReadPath)).size;
  const imported = await readSessionConversationImport(sourceReadPath, {
    tokenizer: dependencies.tokenizer,
    ...(config.chunkPolicy ?? {}),
  });
  const sourceProjections = await createRecallSessionProjectionBaselineFromImport({
    physicalSessionPath: sourceReadPath,
    generationId,
    tokenizer: dependencies.tokenizer,
    ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
    expectedSourceByteSize: sourceByteSize,
    imported,
  });
  const logicalProjections = sourceProjections.filter(
    (projection): projection is LogicalSessionProjection =>
      projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
  );
  const lexicalSource: MaterializedRecallPhysicalSourceGeneration['lexicalSource'] = [];
  const denseInputs: Array<{
    chunk: SessionConversationChunk;
    evidenceOccurrenceId: string;
    physicalSourceIdentity: string;
    logicalSessionOccurrenceId: string;
    projectIdentity: string;
  }> = [];
  const logicalSessionProjections: MaterializedRecallPhysicalSourceGeneration['logicalSessionProjections'] =
    [];
  const logicalSessionOccurrenceIds: string[] = [];

  for (const logicalSession of imported.logicalSessions) {
    const logicalProjection = findLogicalProjection(logicalProjections, logicalSession);
    const logicalSessionOccurrenceId = createRecallLogicalSessionOccurrenceId(
      physicalSource.physicalSourceIdentity,
      logicalSession.sourceLineStart,
    );
    logicalSessionOccurrenceIds.push(logicalSessionOccurrenceId);
    const projectAttribution = await dependencies.resolveProjectIdentity(
      logicalProjection.headerDescriptor.cwd,
    );
    const childIdsByEntryId = new Map<string, string[]>();
    for (const [index, entryId] of logicalSession.entryIds.entries()) {
      const parentEntryId = logicalSession.parentEntryIds[index] ?? null;
      if (parentEntryId !== null) {
        const childIds = childIdsByEntryId.get(parentEntryId) ?? [];
        childIds.push(entryId);
        childIdsByEntryId.set(parentEntryId, childIds);
      }
    }
    const branchLeafIdsByEntryId = createBranchLeafIdsByEntryId(logicalSession);
    const entryAnchorIds: string[] = [];
    for (const descriptor of logicalProjection.entryDescriptors) {
      const entryAnchorId = createRecallEntryAnchorId({
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId,
        entryId: descriptor.entryId,
        sourceLine: descriptor.sourceLine,
        startByte: descriptor.startByte,
        endByte: descriptor.endByte,
      });
      entryAnchorIds.push(entryAnchorId);
      const commonFields = createCommonLexicalFields({
        generationId,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        sessionsRootRelativePath: physicalSource.sessionsRootRelativePath,
        logicalSessionOccurrenceId,
        rawSessionId: logicalSession.sessionId,
        headerSourceLine: logicalSession.sourceLineStart,
        entryAnchorId,
        entryId: descriptor.entryId,
        parentEntryId: descriptor.parentEntryId,
        childEntryIds: childIdsByEntryId.get(descriptor.entryId) ?? [],
        branchPathLeafIds: branchLeafIdsByEntryId.get(descriptor.entryId) ?? [],
        sourceOrder: descriptor.sourceLine,
        entryType: descriptor.entryType,
        timestamp: descriptor.timestamp,
        entryStartByte: descriptor.startByte,
        entryEndByte: descriptor.endByte,
      });
      const anchorRecord = {
        ...commonFields,
        recordKind: 'entry-anchor',
        evidenceOccurrenceId: '',
        documentKind: '',
        evidenceKind: '',
        evidencePart: '',
        isDenseSearchable: false,
        evidenceChecksum: '',
        projectIdentity: projectAttribution?.projectIdentity ?? '',
        sourceLineStart: descriptor.sourceLine,
        sourceLineEnd: descriptor.sourceLine,
        sourceBlockStart: -1,
        sourceBlockEnd: -1,
        characterStart: 0,
        characterEnd: 0,
        tokenStart: 0,
        tokenEnd: 0,
        textRunIndex: -1,
        chunkIndex: -1,
        content: '',
        identifierContent: '',
        recordJson: JSON.stringify({
          recordKind: 'entry-anchor',
          physicalSource,
          logicalSessionOccurrenceId,
          rawSessionId: logicalSession.sessionId,
          projectAttribution,
          descriptor,
          childEntryIds: childIdsByEntryId.get(descriptor.entryId) ?? [],
          branchPathLeafIds: branchLeafIdsByEntryId.get(descriptor.entryId) ?? [],
        }),
      };
      lexicalSource.push({ id: entryAnchorId, fields: anchorRecord });
    }
    logicalSessionProjections.push({
      id: `projection_${logicalSessionOccurrenceId}`,
      fields: {
        schemaVersion: 1,
        generationId,
        projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId,
        projectionJson: JSON.stringify({
          schemaVersion: 1,
          generationId,
          projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
          physicalSource,
          logicalSessionOccurrenceId,
          rawSessionId: logicalSession.sessionId,
          headerSourceLine: logicalSession.sourceLineStart,
          projectAttribution,
          entryAnchorIds,
        }),
      },
    });
  }

  for (const chunk of imported.chunks) {
    const logicalSession = findLogicalSessionForChunk(imported.logicalSessions, chunk);
    const logicalProjection = findLogicalProjection(logicalProjections, logicalSession);
    const descriptor = logicalProjection.entryDescriptors.find(
      ({ entryId }) => entryId === chunk.entryId.value,
    );
    if (descriptor === undefined) {
      throw new Error(
        `Recall physical source generation entry anchor missing for ${physicalSessionPath}:${chunk.entryId.value}`,
      );
    }
    const logicalSessionOccurrenceId = createRecallLogicalSessionOccurrenceId(
      physicalSource.physicalSourceIdentity,
      logicalSession.sourceLineStart,
    );
    const entryAnchorId = createRecallEntryAnchorId({
      physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      logicalSessionOccurrenceId,
      entryId: descriptor.entryId,
      sourceLine: descriptor.sourceLine,
      startByte: descriptor.startByte,
      endByte: descriptor.endByte,
    });
    const evidenceOccurrenceId = createRecallEvidenceOccurrenceId({
      physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      logicalSessionOccurrenceId,
      entryId: chunk.entryId.value,
      evidencePart: chunk.evidencePart,
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
    });
    const projectAttribution = await dependencies.resolveProjectIdentity(chunk.cwd);
    const childEntryIds = logicalSession.entryIds.filter(
      (_, index) => logicalSession.parentEntryIds[index] === chunk.entryId.value,
    );
    const branchPathLeafIds = createBranchLeafIdsByEntryId(logicalSession).get(chunk.entryId.value);
    const commonFields = createCommonLexicalFields({
      generationId,
      physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      sessionsRootRelativePath: physicalSource.sessionsRootRelativePath,
      logicalSessionOccurrenceId,
      rawSessionId: logicalSession.sessionId,
      headerSourceLine: logicalSession.sourceLineStart,
      entryAnchorId,
      entryId: chunk.entryId.value,
      parentEntryId: chunk.parentEntryId?.value ?? null,
      childEntryIds,
      branchPathLeafIds: branchPathLeafIds ?? [],
      sourceOrder: descriptor.sourceLine,
      entryType: descriptor.entryType,
      timestamp: chunk.timestamp,
      entryStartByte: descriptor.startByte,
      entryEndByte: descriptor.endByte,
    });
    const evidence: RecallGenerationLexicalEvidence = {
      evidenceOccurrenceId,
      physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      sessionsRootRelativePath: physicalSource.sessionsRootRelativePath,
      logicalSessionOccurrenceId,
      rawSessionId: logicalSession.sessionId,
      entryAnchorId,
      entryId: chunk.entryId.value,
      parentEntryId: chunk.parentEntryId?.value ?? null,
      evidenceKind: chunk.evidenceKind,
      evidencePart: chunk.evidencePart,
      isDenseSearchable: chunk.isDenseSearchable,
      projectIdentity: projectAttribution?.projectIdentity ?? '',
      sourceLineStart: chunk.sourceLineStart,
      sourceLineEnd: chunk.sourceLineEnd,
      content: chunk.content,
    };
    lexicalSource.push({
      id: evidenceOccurrenceId,
      fields: {
        ...commonFields,
        recordKind: 'evidence',
        evidenceOccurrenceId,
        documentKind: chunk.documentKind,
        evidenceKind: chunk.evidenceKind,
        evidencePart: chunk.evidencePart,
        isDenseSearchable: chunk.isDenseSearchable,
        evidenceChecksum: chunk.checksum,
        projectIdentity: evidence.projectIdentity,
        sourceLineStart: chunk.sourceLineStart,
        sourceLineEnd: chunk.sourceLineEnd,
        sourceBlockStart: chunk.sourceBlockStart ?? -1,
        sourceBlockEnd: chunk.sourceBlockEnd ?? -1,
        characterStart: chunk.characterStart,
        characterEnd: chunk.characterEnd,
        tokenStart: chunk.tokenStart,
        tokenEnd: chunk.tokenEnd,
        textRunIndex: chunk.textRunIndex,
        chunkIndex: chunk.chunkIndex,
        content: chunk.content,
        identifierContent: chunk.content,
        recordJson: JSON.stringify(evidence),
      },
    });
    if (chunk.isDenseSearchable) {
      denseInputs.push({
        chunk,
        evidenceOccurrenceId,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId,
        projectIdentity: evidence.projectIdentity,
      });
    }
  }

  const storedDimensions =
    config.embeddingProfile.storedDimensions ?? config.embeddingProfile.identity.dimensions;
  const dense: RecallGenerationDenseExpectation[] = denseInputs.map((input) => ({
    id: input.evidenceOccurrenceId,
    embeddingInput: input.chunk.content,
    fields: {
      schemaVersion: 1,
      generationId,
      evidenceOccurrenceId: input.evidenceOccurrenceId,
      physicalSourceIdentity: input.physicalSourceIdentity,
      logicalSessionOccurrenceId: input.logicalSessionOccurrenceId,
      embeddingProfileId: config.embeddingProfileId,
      storedDimensions,
      evidenceChecksum: input.chunk.checksum,
      embeddingInputChecksum: createHash('sha256')
        .update(`${config.embeddingProfile.documentInputPrefix}${input.chunk.content}`)
        .digest('hex'),
      vectorChecksum: '',
      projectIdentity: input.projectIdentity,
    },
  }));
  const physicalSessionProjectionId = `projection_${physicalSource.physicalSourceIdentity}`;
  const physicalSessionProjection = {
    id: physicalSessionProjectionId,
    fields: {
      schemaVersion: 1,
      generationId,
      projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
      physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      logicalSessionOccurrenceId: '',
      projectionJson: JSON.stringify({
        schemaVersion: 1,
        generationId,
        projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
        physicalSource,
        sourceByteSize,
        logicalSessionOccurrenceIds,
        expectedMembership: {
          lexicalSource: createRecallPhysicalSourceMembership(lexicalSource.map(({ id }) => id)),
          dense: createRecallPhysicalSourceMembership(dense.map(({ id }) => id)),
          sessionProjection: createRecallPhysicalSourceMembership([
            ...logicalSessionProjections.map(({ id }) => id),
            physicalSessionProjectionId,
          ]),
        },
      }),
    },
  };
  return {
    lexicalSource,
    dense,
    logicalSessionProjections,
    physicalSessionProjection,
    physicalSourceIdentities: [physicalSource.physicalSourceIdentity],
    logicalSessionOccurrenceIds,
  };
}

/** Parses one source-faithful evidence payload fetched through a target read adapter. */
export function parseRecallGenerationLexicalEvidence(
  recordJson: unknown,
): RecallGenerationLexicalEvidence {
  if (typeof recordJson !== 'string') {
    throw new Error('Recall physical source generation evidence record JSON missing');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(recordJson);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall physical source generation evidence record JSON invalid: ${message}`, {
      cause: error,
    });
  }
  try {
    return Value.Parse(recallGenerationLexicalEvidenceSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall physical source generation evidence record invalid: ${message}`, {
      cause: error,
    });
  }
}

/** Searches ordinary lexical evidence in one explicitly named validated inactive generation. */
export async function searchRecallGenerationLexical(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  query: string,
  limit: number,
): Promise<RecallGenerationLexicalEvidence[]> {
  if (!query.trim()) {
    throw new Error('Recall physical source generation lexical query must not be blank');
  }
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) {
    throw new Error('Recall physical source generation lexical limit must be from 1 to 200');
  }
  const generationDirectory = join(config.generationRootDirectory, generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  if (existsSync(paths.recoveryRecordPath)) {
    throw new Error(`Recall coherent generation recovery required for ${generationId}`);
  }
  const expectedManifest = createExpectedRecallPhysicalSourceManifest(config, generationId);
  const { manifest } = await readRecallGenerationManifest(paths.manifestPath);
  assertRecallGenerationManifestCompatible(manifest, expectedManifest, paths.manifestPath);
  await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  const collection = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  try {
    const documents = await collection.query({
      fieldName: 'content',
      fts: { matchString: query },
      filter: "recordKind = 'evidence'",
      topk: limit,
      outputFields: ['recordJson'],
      includeVector: false,
      params: { indexType: ZVecIndexType.FTS, defaultOperator: 'OR' },
    });
    return documents.map(({ fields }) => parseRecallGenerationLexicalEvidence(fields.recordJson));
  } finally {
    collection.closeSync();
  }
}

async function listPhysicalSourceRecordIds(
  collection: ZVecCollection,
  physicalSourceIdentity: string,
): Promise<string[]> {
  if (collection.stats.docCount === 0) {
    return [];
  }
  const documents = await collection.query({
    filter: `physicalSourceIdentity = '${physicalSourceIdentity}'`,
    topk: collection.stats.docCount,
    outputFields: [],
    includeVector: false,
  });
  return documents.map(({ id }) => id);
}

/** Deletes rows joined to exactly one physical source, with dense evidence removed first. */
export async function deleteRecallGenerationPhysicalSource(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  physicalSourceIdentity: string,
): Promise<void> {
  const paths = createRecallGenerationComponentPaths(
    join(config.generationRootDirectory, generationId),
  );
  const dense = ZVecOpen(paths.denseStorePath);
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
  const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
  try {
    const [denseIds, lexicalSourceIds, projectionIds] = await Promise.all([
      listPhysicalSourceRecordIds(dense, physicalSourceIdentity),
      listPhysicalSourceRecordIds(lexicalSource, physicalSourceIdentity),
      listPhysicalSourceRecordIds(sessionProjection, physicalSourceIdentity),
    ]);
    await writeFile(
      paths.recoveryRecordPath,
      `${JSON.stringify(
        {
          version: 1,
          generationId,
          operation: 'delete-physical-source',
          physicalSourceIdentity,
          denseIds,
          lexicalSourceIds,
          projectionIds,
        },
        null,
        2,
      )}\n`,
      { encoding: 'utf8', flag: 'wx' },
    );
    if (denseIds.length > 0) {
      assertCheckedZvecStatuses('dense deletion', denseIds, dense.deleteSync(denseIds));
    }
    if (lexicalSourceIds.length > 0) {
      assertCheckedZvecStatuses(
        'lexical/source deletion',
        lexicalSourceIds,
        lexicalSource.deleteSync(lexicalSourceIds),
      );
    }
    if (projectionIds.length > 0) {
      assertCheckedZvecStatuses(
        'session projection deletion',
        projectionIds,
        sessionProjection.deleteSync(projectionIds),
      );
    }
  } finally {
    dense.closeSync();
    lexicalSource.closeSync();
    sessionProjection.closeSync();
  }
  await rm(paths.recoveryRecordPath);
}
