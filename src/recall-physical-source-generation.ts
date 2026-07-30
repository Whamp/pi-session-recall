import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecIndexType, ZVecOpen, type ZVecCollection, type ZVecStatus } from '@zvec/zvec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallSessionProjectionKind } from './enums.js';
import { createRecallSessionProjectionBaseline } from './create-recall-session-projection-baseline.js';
import type {
  OpenedValidatedRecallGeneration,
  RecallCoherentGenerationConfig,
} from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  createRecallGenerationManifest,
  readRecallGenerationManifest,
  writeRecallGenerationManifest,
} from './recall-generation-manifest.js';
import {
  createEmptyRecallGenerationStores,
  createRecallGenerationComponentPaths,
  createRecallGenerationStoreContracts,
  validateRecallGenerationStores,
} from './recall-generation-stores.js';
import {
  createRecallGenerationValidationReceipt,
  readRecallGenerationValidationReceipt,
  writeRecallGenerationValidationReceipt,
} from './recall-generation-validation-receipt.js';
import { createRecallActiveGenerationPointer } from './recall-generation-state.js';
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
  assertRepeatableStoredRecallEmbeddings,
  createStoredRecallEmbedding,
} from './recall-stored-embedding.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
  type SessionConversationLogicalSession,
} from './session-conversation-index.js';

/** Physical files selected for one inactive lexical-only target generation. */
export interface CreateRecallGenerationFromPhysicalSourcesOptions {
  generationId: string;
  physicalSessionPaths: readonly string[];
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

/** Runtime dependencies needed to materialize source identities at the canonical import boundary. */
export interface RecallPhysicalSourceGenerationDependencies {
  tokenizer: ConversationTextTokenizer;
  embeddingProvider: RecallEmbeddingProvider;
  resolveProjectIdentity(workingDirectory: string): Promise<ResolvedProjectIdentity | null>;
}

interface RecallGenerationDenseRow {
  id: string;
  fields: Record<string, unknown>;
  vectors: { embedding: number[] };
}

interface MaterializedRecallGenerationRows {
  lexicalSource: Array<{ id: string; fields: Record<string, unknown> }>;
  dense: RecallGenerationDenseRow[];
  sessionProjection: Array<{ id: string; fields: Record<string, unknown> }>;
  physicalSourceIdentities: string[];
  logicalSessionOccurrenceIds: string[];
}

function createExpectedManifest(
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

async function materializePhysicalSourceRows(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  physicalSessionPath: string,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<MaterializedRecallGenerationRows> {
  const physicalSource = resolveRecallPhysicalSourceIdentity(
    config.sessionsDirectory,
    physicalSessionPath,
  );
  const sourceByteSize = (await stat(physicalSessionPath)).size;
  const [imported, sourceProjections] = await Promise.all([
    readSessionConversationImport(physicalSessionPath, {
      tokenizer: dependencies.tokenizer,
      ...(config.chunkPolicy ?? {}),
    }),
    createRecallSessionProjectionBaseline({
      physicalSessionPath,
      generationId,
      tokenizer: dependencies.tokenizer,
      ...(config.chunkPolicy ?? {}),
      expectedSourceByteSize: sourceByteSize,
    }),
  ]);
  const logicalProjections = sourceProjections.filter(
    (projection): projection is LogicalSessionProjection =>
      projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
  );
  const lexicalSource: MaterializedRecallGenerationRows['lexicalSource'] = [];
  const denseInputs: Array<{
    chunk: SessionConversationChunk;
    evidenceOccurrenceId: string;
    physicalSourceIdentity: string;
    logicalSessionOccurrenceId: string;
    projectIdentity: string;
  }> = [];
  const sessionProjection: MaterializedRecallGenerationRows['sessionProjection'] = [];
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
    sessionProjection.push({
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

  const dense: RecallGenerationDenseRow[] = [];
  if (denseInputs.length > 0) {
    const nativeVectors = await dependencies.embeddingProvider.embedDocuments(
      denseInputs.map(({ chunk }) => chunk.content),
    );
    if (nativeVectors.length !== denseInputs.length) {
      throw new Error(
        `Recall physical source generation document embedding count mismatch at ${physicalSessionPath}: expected ${denseInputs.length}, received ${nativeVectors.length}`,
      );
    }
    const nativeDimensions = config.embeddingProfile.identity.dimensions;
    const storedDimensions = config.embeddingProfile.storedDimensions ?? nativeDimensions;
    for (const [index, input] of denseInputs.entries()) {
      const nativeVector = nativeVectors[index];
      if (nativeVector === undefined) {
        throw new Error(
          `Recall physical source generation document embedding missing at ${physicalSessionPath}:${input.chunk.sourceLineStart}`,
        );
      }
      const embedding = createStoredRecallEmbedding(nativeVector, {
        nativeDimensions,
        storedDimensions,
        source: `${physicalSessionPath}:${input.chunk.sourceLineStart}:${input.evidenceOccurrenceId}`,
      });
      const embeddingInputChecksum = createHash('sha256')
        .update(`${config.embeddingProfile.documentInputPrefix}${input.chunk.content}`)
        .digest('hex');
      const vectorChecksum = createHash('sha256')
        .update(Buffer.from(new Float32Array(embedding).buffer))
        .digest('hex');
      dense.push({
        id: input.evidenceOccurrenceId,
        vectors: { embedding },
        fields: {
          schemaVersion: 1,
          generationId,
          evidenceOccurrenceId: input.evidenceOccurrenceId,
          physicalSourceIdentity: input.physicalSourceIdentity,
          logicalSessionOccurrenceId: input.logicalSessionOccurrenceId,
          embeddingProfileId: config.embeddingProfileId,
          storedDimensions,
          evidenceChecksum: input.chunk.checksum,
          embeddingInputChecksum,
          vectorChecksum,
          projectIdentity: input.projectIdentity,
        },
      });
    }
  }

  sessionProjection.push({
    id: `projection_${physicalSource.physicalSourceIdentity}`,
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
      }),
    },
  });
  return {
    lexicalSource,
    dense,
    sessionProjection,
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

/** Creates and completely validates one inactive mixed lexical and dense generation. */
export async function createRecallGenerationFromPhysicalSources(
  config: Readonly<RecallCoherentGenerationConfig>,
  options: Readonly<CreateRecallGenerationFromPhysicalSourcesOptions>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<OpenedValidatedRecallGeneration> {
  createRecallActiveGenerationPointer(options.generationId);
  if (options.physicalSessionPaths.length === 0) {
    throw new Error('Recall physical source generation requires at least one physical session');
  }
  await mkdir(config.generationRootDirectory, { recursive: true });
  const generationDirectory = join(config.generationRootDirectory, options.generationId);
  await mkdir(generationDirectory);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  try {
    const manifest = createExpectedManifest(config, options.generationId);
    if (config.embeddingProfile.canary) {
      const [firstNativeCanary, repeatedNativeCanary] = await Promise.all([
        dependencies.embeddingProvider.embedQuery(config.embeddingProfile.canary.query),
        dependencies.embeddingProvider.embedQuery(config.embeddingProfile.canary.query),
      ]);
      const canaryOptions = {
        nativeDimensions: manifest.embeddingProfile.nativeDimensions,
        storedDimensions: manifest.embeddingProfile.storedDimensions,
        source: `generation ${options.generationId} query canary`,
      };
      const firstStoredCanary = createStoredRecallEmbedding(firstNativeCanary, canaryOptions);
      const repeatedStoredCanary = createStoredRecallEmbedding(repeatedNativeCanary, canaryOptions);
      assertRepeatableStoredRecallEmbeddings(firstStoredCanary, repeatedStoredCanary, {
        minimumCosineSimilarity: config.embeddingProfile.canary.minimumRepeatCosineSimilarity,
        source: `generation ${options.generationId} query canary`,
      });
    }
    const manifestFingerprint = await writeRecallGenerationManifest(paths.manifestPath, manifest);
    const contracts = createRecallGenerationStoreContracts(
      options.generationId,
      manifest.embeddingProfile.storedDimensions,
    );
    createEmptyRecallGenerationStores(paths, contracts);
    const materializedSources = await Promise.all(
      options.physicalSessionPaths.map((physicalSessionPath) =>
        materializePhysicalSourceRows(
          config,
          options.generationId,
          physicalSessionPath,
          dependencies,
        ),
      ),
    );
    const rows: MaterializedRecallGenerationRows = {
      lexicalSource: materializedSources.flatMap(({ lexicalSource }) => lexicalSource),
      dense: materializedSources.flatMap(({ dense }) => dense),
      sessionProjection: materializedSources.flatMap(({ sessionProjection }) => sessionProjection),
      physicalSourceIdentities: materializedSources.flatMap(
        ({ physicalSourceIdentities }) => physicalSourceIdentities,
      ),
      logicalSessionOccurrenceIds: materializedSources.flatMap(
        ({ logicalSessionOccurrenceIds }) => logicalSessionOccurrenceIds,
      ),
    };
    if (
      new Set(rows.physicalSourceIdentities).size !== rows.physicalSourceIdentities.length ||
      new Set(rows.lexicalSource.map(({ id }) => id)).size !== rows.lexicalSource.length ||
      new Set(rows.dense.map(({ id }) => id)).size !== rows.dense.length ||
      new Set(rows.sessionProjection.map(({ id }) => id)).size !== rows.sessionProjection.length
    ) {
      throw new Error('Recall physical source generation contains duplicate source identities');
    }
    const lexicalCollection = ZVecOpen(paths.lexicalSourceStorePath);
    const denseCollection = ZVecOpen(paths.denseStorePath);
    const projectionCollection = ZVecOpen(paths.sessionProjectionStorePath);
    try {
      assertCheckedZvecStatuses(
        'lexical/source write',
        rows.lexicalSource.map(({ id }) => id),
        lexicalCollection.upsertSync(rows.lexicalSource),
      );
      assertCheckedZvecStatuses(
        'dense write',
        rows.dense.map(({ id }) => id),
        denseCollection.upsertSync(rows.dense),
      );
      assertCheckedZvecStatuses(
        'session projection write',
        rows.sessionProjection.map(({ id }) => id),
        projectionCollection.upsertSync(rows.sessionProjection),
      );
    } finally {
      lexicalCollection.closeSync();
      denseCollection.closeSync();
      projectionCollection.closeSync();
    }
    const expectedRecordIds = {
      lexicalSource: rows.lexicalSource.map(({ id }) => id).toSorted(),
      dense: rows.dense.map(({ id }) => id).toSorted(),
      sessionProjection: rows.sessionProjection.map(({ id }) => id).toSorted(),
    };
    const storeCounts = validateRecallGenerationStores(
      paths,
      contracts,
      options.generationId,
      expectedRecordIds,
    );
    const startingSnapshotFingerprint = createHash('sha256')
      .update(
        JSON.stringify({
          physicalSourceIdentities: rows.physicalSourceIdentities.toSorted(),
          logicalSessionOccurrenceIds: rows.logicalSessionOccurrenceIds.toSorted(),
          expectedRecordIds,
        }),
      )
      .digest('hex');
    const receipt = createRecallGenerationValidationReceipt({
      generationId: options.generationId,
      manifestFingerprint,
      membership: {
        startingSnapshotFingerprint,
        physicalSourceCount: rows.physicalSourceIdentities.length,
        logicalSessionOccurrenceCount: rows.logicalSessionOccurrenceIds.length,
        lexicalSourceRecordIds: expectedRecordIds.lexicalSource,
        denseRecordIds: expectedRecordIds.dense,
        sessionProjectionRecordIds: expectedRecordIds.sessionProjection,
      },
      validatedAtEpochMilliseconds: (config.nowEpochMilliseconds ?? Date.now)(),
    });
    await writeRecallGenerationValidationReceipt(paths.validationReceiptPath, receipt);
    return {
      generationId: options.generationId,
      generationDirectory,
      manifestPath: paths.manifestPath,
      validationReceiptPath: paths.validationReceiptPath,
      manifestFingerprint,
      startingSnapshotFingerprint,
      storeCounts,
    };
  } catch (error) {
    await rm(generationDirectory, { recursive: true, force: true });
    throw error;
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
  const expectedManifest = createExpectedManifest(config, generationId);
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
