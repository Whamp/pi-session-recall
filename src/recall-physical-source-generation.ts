import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecIndexType, ZVecOpen } from '@zvec/zvec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallProjectionEncodingStatus, RecallSessionProjectionKind } from './enums.js';
import { createRecallSessionProjectionBaselineFromImport } from './create-recall-session-projection-baseline.js';
import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  createRecallGenerationManifest,
  readRecallGenerationManifest,
} from './recall-generation-manifest.js';
import { createRecallPhysicalSourceStoreMembership } from './recall-generation-physical-projection.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import {
  createRecallEntryAnchorId,
  createRecallEvidenceOccurrenceId,
  createRecallLogicalSessionOccurrenceId,
  resolveRecallPhysicalSourceIdentity,
} from './recall-source-identity.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  encodeRecallSessionProjection,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallSessionProjection,
} from './recall-session-projection.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
  type SessionConversationLogicalSession,
} from './session-conversation-index.js';
import {
  deserializeStoredConversationChunk,
  serializeStoredConversationChunk,
} from './zvec-conversation-store.js';

/** Durable source progress emitted after a projection reopens or an invalid-source skip is recorded. */
export interface RecallFixedSnapshotPhysicalSourceCheckpoint {
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  completedPhysicalSourceCount: number;
  totalPhysicalSourceCount: number;
}

/** Fixed physical-source snapshot selected for one resumable inactive target generation. */
export interface CreateRecallGenerationFromPhysicalSourcesOptions {
  generationId: string;
  physicalSessionPaths: readonly string[];
  validatedVectorSourceGenerationId?: string;
  resumeExistingGeneration?: boolean;
  signal?: AbortSignal;
  onPhysicalSourceCheckpoint?(
    checkpoint: Readonly<RecallFixedSnapshotPhysicalSourceCheckpoint>,
  ): void;
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
const recallGenerationEvidenceRecordSchema = Type.Object(
  {
    evidence: recallGenerationLexicalEvidenceSchema,
    searchDocument: Type.Record(Type.String(), Type.Unknown()),
  },
  { additionalProperties: false },
);

/** Runtime dependencies needed to materialize and fault-probe a fixed source snapshot. */
export interface RecallPhysicalSourceGenerationDependencies {
  tokenizer: ConversationTextTokenizer;
  embeddingProvider: RecallEmbeddingProvider;
  resolveProjectIdentity(workingDirectory: string): Promise<ResolvedProjectIdentity | null>;
  fixedSnapshotBuildFault?: (
    stage:
      | 'after-snapshot-capture'
      | 'after-dense-write'
      | 'after-store-close'
      | 'before-validation-receipt',
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

function createRecallProjectIdentityDigest(projectIdentity: string): string {
  return projectIdentity ? createHash('sha256').update(projectIdentity).digest('hex') : '';
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

function encodeTargetIngestionProjection(projection: RecallSessionProjection) {
  const encoded = encodeRecallSessionProjection(projection);
  if (encoded.status !== RecallProjectionEncodingStatus.ENCODED) {
    throw new Error(
      `Recall physical source generation projection exceeds bounded payload: ${projection.projectionId}`,
    );
  }
  return encoded.payload;
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
  evidenceOccurrenceIds: readonly string[];
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
    evidenceOccurrenceIds: [...options.evidenceOccurrenceIds],
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
  const [sourceByteSize, physicalSourceMetadata] = await Promise.all([
    stat(sourceReadPath).then(({ size }) => size),
    stat(physicalSessionPath, { bigint: true }),
  ]);
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
  const physicalProjection = sourceProjections.find(
    (projection): projection is PhysicalSessionProjection =>
      projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
  );
  if (physicalProjection === undefined) {
    throw new Error(
      `Recall physical source generation physical projection missing for ${physicalSessionPath}`,
    );
  }
  const targetPhysicalProjection: PhysicalSessionProjection = {
    ...physicalProjection,
    sourcePath: physicalSessionPath,
    sourceDevice: physicalSourceMetadata.dev.toString(),
    sourceInode: physicalSourceMetadata.ino.toString(),
  };
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
  const projectAttributionBySessionOrigin = new Map<
    string,
    Promise<ResolvedProjectIdentity | null>
  >();
  function resolveSessionProjectAttribution(
    sessionOrigin: string,
  ): Promise<ResolvedProjectIdentity | null> {
    const existing = projectAttributionBySessionOrigin.get(sessionOrigin);
    if (existing !== undefined) {
      return existing;
    }
    const resolution = dependencies.resolveProjectIdentity(sessionOrigin);
    projectAttributionBySessionOrigin.set(sessionOrigin, resolution);
    return resolution;
  }

  const occurrenceIdByLogicalChunkId = new Map<string, string>();
  const occurrenceIdsByLogicalEntryId = new Map<string, string[]>();
  for (const chunk of imported.chunks) {
    const logicalSession = findLogicalSessionForChunk(imported.logicalSessions, chunk);
    const logicalSessionOccurrenceId = createRecallLogicalSessionOccurrenceId(
      physicalSource.physicalSourceIdentity,
      logicalSession.sourceLineStart,
    );
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
    occurrenceIdByLogicalChunkId.set(
      `${logicalSessionOccurrenceId}:${chunk.id}`,
      evidenceOccurrenceId,
    );
    const logicalEntryIdentity = `${logicalSessionOccurrenceId}:${chunk.entryId.value}`;
    const entryOccurrenceIds = occurrenceIdsByLogicalEntryId.get(logicalEntryIdentity) ?? [];
    entryOccurrenceIds.push(evidenceOccurrenceId);
    occurrenceIdsByLogicalEntryId.set(logicalEntryIdentity, entryOccurrenceIds);
  }

  for (const logicalSession of imported.logicalSessions) {
    const logicalProjection = findLogicalProjection(logicalProjections, logicalSession);
    const logicalSessionOccurrenceId = createRecallLogicalSessionOccurrenceId(
      physicalSource.physicalSourceIdentity,
      logicalSession.sourceLineStart,
    );
    logicalSessionOccurrenceIds.push(logicalSessionOccurrenceId);
    const projectAttribution = await resolveSessionProjectAttribution(
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
        evidenceOccurrenceIds:
          occurrenceIdsByLogicalEntryId.get(
            `${logicalSessionOccurrenceId}:${descriptor.entryId}`,
          ) ?? [],
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
        projectIdentityDigest: createRecallProjectIdentityDigest(
          projectAttribution?.projectIdentity ?? '',
        ),
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
          evidenceOccurrenceIds:
            occurrenceIdsByLogicalEntryId.get(
              `${logicalSessionOccurrenceId}:${descriptor.entryId}`,
            ) ?? [],
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
          ingestionProjectionPayload: encodeTargetIngestionProjection(logicalProjection),
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
    const evidenceOccurrenceId = occurrenceIdByLogicalChunkId.get(
      `${logicalSessionOccurrenceId}:${chunk.id}`,
    );
    if (evidenceOccurrenceId === undefined) {
      throw new Error(
        `Recall physical source generation occurrence missing for ${physicalSessionPath}:${chunk.id}`,
      );
    }
    const projectAttribution = await resolveSessionProjectAttribution(chunk.cwd);
    const childEntryIds = logicalSession.entryIds.filter((entryId, index) => {
      void entryId;
      return logicalSession.parentEntryIds[index] === chunk.entryId.value;
    });
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
      evidenceOccurrenceIds:
        occurrenceIdsByLogicalEntryId.get(`${logicalSessionOccurrenceId}:${descriptor.entryId}`) ??
        [],
      sourceOrder: descriptor.sourceLine,
      entryType: descriptor.entryType,
      timestamp: chunk.timestamp,
      entryStartByte: descriptor.startByte,
      entryEndByte: descriptor.endByte,
    });
    const mapSiblingOccurrenceId = (chunkId: string): string => {
      const occurrenceId = occurrenceIdByLogicalChunkId.get(
        `${logicalSessionOccurrenceId}:${chunkId}`,
      );
      if (occurrenceId === undefined) {
        throw new Error(
          `Recall physical source generation sibling occurrence missing for ${physicalSessionPath}:${chunkId}`,
        );
      }
      return occurrenceId;
    };
    const targetSearchChunk: SessionConversationChunk = {
      ...chunk,
      id: evidenceOccurrenceId,
      sessionPath: physicalSessionPath,
      physicalSessionProjectionId: physicalSource.physicalSourceIdentity,
      projectAttribution,
      siblingIds: chunk.siblingIds.map(mapSiblingOccurrenceId),
      previousSiblingId:
        chunk.previousSiblingId === null ? null : mapSiblingOccurrenceId(chunk.previousSiblingId),
      nextSiblingId:
        chunk.nextSiblingId === null ? null : mapSiblingOccurrenceId(chunk.nextSiblingId),
    };
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
        projectIdentityDigest: createRecallProjectIdentityDigest(evidence.projectIdentity),
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
        recordJson: JSON.stringify({
          evidence,
          searchDocument: serializeStoredConversationChunk(targetSearchChunk),
        }),
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
      projectIdentityDigest: createRecallProjectIdentityDigest(input.projectIdentity),
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
          lexicalSource: createRecallPhysicalSourceStoreMembership(
            lexicalSource.map(({ id }) => id),
          ),
          dense: createRecallPhysicalSourceStoreMembership(dense.map(({ id }) => id)),
          sessionProjection: createRecallPhysicalSourceStoreMembership([
            ...logicalSessionProjections.map(({ id }) => id),
            physicalSessionProjectionId,
          ]),
        },
        ingestionProjectionPayload: encodeTargetIngestionProjection(targetPhysicalProjection),
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

function parseRecallGenerationEvidenceRecord(recordJson: unknown) {
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
    return Value.Parse(recallGenerationEvidenceRecordSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall physical source generation evidence record invalid: ${message}`, {
      cause: error,
    });
  }
}

/** Parses one source-faithful evidence payload fetched through a target read adapter. */
export function parseRecallGenerationLexicalEvidence(
  recordJson: unknown,
): RecallGenerationLexicalEvidence {
  return parseRecallGenerationEvidenceRecord(recordJson).evidence;
}

/** Restores one mature search document with its stable evidence occurrence ID. */
export function parseRecallGenerationSearchDocument(
  evidenceOccurrenceId: string,
  recordJson: unknown,
): SessionConversationChunk {
  const record = parseRecallGenerationEvidenceRecord(recordJson);
  if (record.evidence.evidenceOccurrenceId !== evidenceOccurrenceId) {
    throw new Error(
      `Recall physical source generation occurrence mismatch: expected ${evidenceOccurrenceId}, received ${record.evidence.evidenceOccurrenceId}`,
    );
  }
  return deserializeStoredConversationChunk(evidenceOccurrenceId, record.searchDocument);
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
