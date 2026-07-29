import { createHash } from 'node:crypto';

import { RecallSessionProjectionKind, type SessionImportFormat } from '../enums.js';
import { normalizeConversationTextForEmbedding } from '../embedding-vector-cache.js';
import { createRecallSessionProjectionBaseline } from '../create-recall-session-projection-baseline.js';
import {
  createLogicalSessionOccurrenceId,
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  type RecallSessionProjection,
} from '../recall-session-projection.js';
import type { RecallChunkPolicy } from '../recall-chunk-policy.js';
import type { ResolvedProjectIdentity } from '../resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
  type SessionConversationImport,
  type SessionConversationLogicalSession,
} from '../session-conversation-index.js';
import type {
  RecallBenchmarkEntryAnchor,
  RecallBenchmarkEvidenceOccurrence,
} from './recall-benchmark-split-store.js';

/** Timed production seams used to prepare one sampled physical source. */
export interface RecallBenchmarkSourcePreparationTimings {
  importMilliseconds: number;
  projectionMilliseconds: number;
  projectIdentityMilliseconds: number;
  rowMaterializationMilliseconds: number;
}

/** All split-store rows prepared from one frozen physical source before database writes. */
export interface PreparedRecallBenchmarkSource {
  format: SessionImportFormat;
  physicalSourceId: string;
  physicalSessionProjectionId: string;
  entryAnchors: RecallBenchmarkEntryAnchor[];
  evidenceOccurrences: RecallBenchmarkEvidenceOccurrence[];
  denseOccurrences: RecallBenchmarkEvidenceOccurrence[];
  projections: RecallSessionProjection[];
  timings: RecallBenchmarkSourcePreparationTimings;
}

function digestBase64Url(domain: string, parts: readonly string[]): string {
  return createHash('sha256')
    .update(JSON.stringify([domain, ...parts]))
    .digest('base64url');
}

function digestHex(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

/** Identifies one physical source by its stable path relative to the configured sessions root. */
export function createRecallBenchmarkPhysicalSourceId(relativePath: string): string {
  return `physical_source_${digestBase64Url('physical_source_path_v1', [relativePath])}`;
}

function findLogicalSessionForChunk(
  chunk: SessionConversationChunk,
  logicalSessions: readonly SessionConversationLogicalSession[],
): SessionConversationLogicalSession {
  const matches = logicalSessions.filter(
    ({ sourceLineStart, sourceLineEnd }) =>
      chunk.sourceLineStart >= sourceLineStart && chunk.sourceLineStart <= sourceLineEnd,
  );
  const logicalSession = matches[0];
  if (matches.length !== 1 || !logicalSession) {
    throw new Error('Recall no-cache benchmark chunk has ambiguous logical-session membership');
  }
  return logicalSession;
}

function createEvidenceOccurrences(options: {
  imported: SessionConversationImport;
  physicalSourceId: string;
  physicalSessionProjectionId: string;
  projectIdentityByOrigin: ReadonlyMap<string, ResolvedProjectIdentity | null>;
}): RecallBenchmarkEvidenceOccurrence[] {
  return options.imported.chunks.map((sourceChunk) => {
    const logicalSession = findLogicalSessionForChunk(
      sourceChunk,
      options.imported.logicalSessions,
    );
    const logicalSessionId = createLogicalSessionOccurrenceId(
      logicalSession.sessionId,
      logicalSession.sourceLineStart,
    );
    const chunk: SessionConversationChunk = {
      ...sourceChunk,
      physicalSessionProjectionId: options.physicalSessionProjectionId,
      projectAttribution: options.projectIdentityByOrigin.get(sourceChunk.cwd) ?? null,
    };
    return {
      occurrenceId: `occurrence_${digestBase64Url('evidence_occurrence_v1', [
        options.physicalSourceId,
        logicalSessionId,
        chunk.id,
        String(chunk.sourceLineStart),
        String(chunk.sourceBlockStart ?? -1),
        String(chunk.characterStart),
      ])}`,
      logicalSessionId,
      physicalSessionProjectionId: options.physicalSessionProjectionId,
      embeddingInputChecksum: digestHex(normalizeConversationTextForEmbedding(chunk.content)),
      chunk,
    };
  });
}

function createEntryAnchors(options: {
  imported: SessionConversationImport;
  evidenceOccurrences: readonly RecallBenchmarkEvidenceOccurrence[];
  physicalSourceId: string;
  physicalSessionProjectionId: string;
}): RecallBenchmarkEntryAnchor[] {
  const occurrencesByLogicalEntry = new Map<string, RecallBenchmarkEvidenceOccurrence[]>();
  for (const occurrence of options.evidenceOccurrences) {
    const key = `${occurrence.logicalSessionId}\0${occurrence.chunk.entryId.value}`;
    const existing = occurrencesByLogicalEntry.get(key) ?? [];
    existing.push(occurrence);
    occurrencesByLogicalEntry.set(key, existing);
  }

  const anchors: RecallBenchmarkEntryAnchor[] = [];
  for (const logicalSession of options.imported.logicalSessions) {
    const logicalSessionId = createLogicalSessionOccurrenceId(
      logicalSession.sessionId,
      logicalSession.sourceLineStart,
    );
    const childEntryIdsByParentId = new Map<string, string[]>();
    for (const [index, entryId] of logicalSession.entryIds.entries()) {
      const parentEntryId = logicalSession.parentEntryIds[index];
      if (!parentEntryId) {
        continue;
      }
      const children = childEntryIdsByParentId.get(parentEntryId) ?? [];
      children.push(entryId);
      childEntryIdsByParentId.set(parentEntryId, children);
    }
    for (const [index, entryId] of logicalSession.entryIds.entries()) {
      const occurrenceKey = `${logicalSessionId}\0${entryId}`;
      const occurrences = occurrencesByLogicalEntry.get(occurrenceKey) ?? [];
      const branchPathLeafIds = [
        ...new Set(
          occurrences.flatMap(({ chunk }) => chunk.branchPathLeafIds.map((id) => id.value)),
        ),
      ].toSorted();
      const sourceLineStart = Math.min(
        ...occurrences.map(({ chunk }) => chunk.sourceLineStart),
        logicalSession.sourceLineStart,
      );
      const sourceLineEnd = Math.max(
        ...occurrences.map(({ chunk }) => chunk.sourceLineEnd),
        logicalSession.sourceLineStart,
      );
      const parentEntryId = logicalSession.parentEntryIds[index] ?? '';
      const childEntryIds = [...(childEntryIdsByParentId.get(entryId) ?? [])].toSorted();
      const occurrenceIds = occurrences.map(({ occurrenceId }) => occurrenceId).toSorted();
      const anchorIdentity = [options.physicalSourceId, logicalSessionId, entryId];
      anchors.push({
        anchorId: `entry_anchor_${digestBase64Url('entry_anchor_v1', anchorIdentity)}`,
        checksum: digestHex(
          JSON.stringify({
            anchorIdentity,
            parentEntryId,
            childEntryIds,
            branchPathLeafIds,
            occurrenceIds,
            sourceLineStart,
            sourceLineEnd,
          }),
        ),
        physicalSessionProjectionId: options.physicalSessionProjectionId,
        logicalSessionId,
        rawSessionId: logicalSession.sessionId,
        sessionPath: occurrences[0]?.chunk.sessionPath ?? '',
        entryId,
        parentEntryId,
        childEntryIds,
        branchPathLeafIds,
        occurrenceIds,
        sourceLineStart,
        sourceLineEnd,
      });
    }
  }
  return anchors;
}

function retargetSessionProjections(options: {
  projections: readonly RecallSessionProjection[];
  physicalSourceId: string;
  generationId: string;
}): RecallSessionProjection[] {
  const physicalSessionProjectionId = createPhysicalSessionProjectionId(options.physicalSourceId);
  return options.projections.map((projection): RecallSessionProjection => {
    const markerCheckpoint = {
      ...projection.markerCheckpoint,
      generationId: options.generationId,
    };
    if (projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION) {
      return {
        ...projection,
        projectionId: physicalSessionProjectionId,
        generationId: options.generationId,
        physicalSessionId: options.physicalSourceId,
        markerCheckpoint,
      };
    }
    return {
      ...projection,
      projectionId: createLogicalSessionProjectionId(
        options.physicalSourceId,
        projection.logicalSessionId,
      ),
      generationId: options.generationId,
      physicalSessionId: options.physicalSourceId,
      physicalProjectionId: physicalSessionProjectionId,
      markerCheckpoint,
    };
  });
}

async function resolveSourceProjectIdentities(
  imported: SessionConversationImport,
  resolveProjectIdentity: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>,
): Promise<Map<string, ResolvedProjectIdentity | null>> {
  const origins = [...new Set(imported.chunks.map(({ cwd }) => cwd).filter(Boolean))];
  return new Map(
    await Promise.all(
      origins.map(
        async (origin): Promise<[string, ResolvedProjectIdentity | null]> => [
          origin,
          await resolveProjectIdentity(origin),
        ],
      ),
    ),
  );
}

/** Runs production import and projection preparation once for one benchmark lane and source. */
export async function prepareRecallBenchmarkSource(options: {
  physicalSessionPath: string;
  relativePath: string;
  expectedSourceBytes: number;
  generationId: string;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy: RecallChunkPolicy;
  resolveProjectIdentity: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
}): Promise<PreparedRecallBenchmarkSource> {
  const importStartedAt = performance.now();
  const imported = await readSessionConversationImport(options.physicalSessionPath, {
    tokenizer: options.tokenizer,
    ...options.chunkPolicy,
  });
  const importMilliseconds = performance.now() - importStartedAt;

  const projectionStartedAt = performance.now();
  const sourceProjections = await createRecallSessionProjectionBaseline({
    physicalSessionPath: options.physicalSessionPath,
    generationId: options.generationId,
    tokenizer: options.tokenizer,
    chunkPolicy: options.chunkPolicy,
    expectedSourceByteSize: options.expectedSourceBytes,
  });
  const projectionMilliseconds = performance.now() - projectionStartedAt;

  const projectIdentityStartedAt = performance.now();
  const projectIdentityByOrigin = await resolveSourceProjectIdentities(
    imported,
    options.resolveProjectIdentity,
  );
  const projectIdentityMilliseconds = performance.now() - projectIdentityStartedAt;

  const rowMaterializationStartedAt = performance.now();
  const physicalSourceId = createRecallBenchmarkPhysicalSourceId(options.relativePath);
  const physicalSessionProjectionId = createPhysicalSessionProjectionId(physicalSourceId);
  const evidenceOccurrences = createEvidenceOccurrences({
    imported,
    physicalSourceId,
    physicalSessionProjectionId,
    projectIdentityByOrigin,
  });
  const entryAnchors = createEntryAnchors({
    imported,
    evidenceOccurrences,
    physicalSourceId,
    physicalSessionProjectionId,
  });
  const projections = retargetSessionProjections({
    projections: sourceProjections,
    physicalSourceId,
    generationId: options.generationId,
  });
  const rowMaterializationMilliseconds = performance.now() - rowMaterializationStartedAt;

  return {
    format: imported.format,
    physicalSourceId,
    physicalSessionProjectionId,
    entryAnchors,
    evidenceOccurrences,
    denseOccurrences: evidenceOccurrences.filter(({ chunk }) => chunk.isDenseSearchable),
    projections,
    timings: {
      importMilliseconds,
      projectionMilliseconds,
      projectIdentityMilliseconds,
      rowMaterializationMilliseconds,
    },
  };
}

/** Recreates stable evidence occurrence identities without projection or project metadata work. */
export async function readRecallBenchmarkEvidenceOccurrences(options: {
  physicalSessionPath: string;
  relativePath: string;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy: RecallChunkPolicy;
}): Promise<{
  format: SessionImportFormat;
  evidenceOccurrences: RecallBenchmarkEvidenceOccurrence[];
  denseOccurrences: RecallBenchmarkEvidenceOccurrence[];
}> {
  const imported = await readSessionConversationImport(options.physicalSessionPath, {
    tokenizer: options.tokenizer,
    ...options.chunkPolicy,
  });
  const physicalSourceId = createRecallBenchmarkPhysicalSourceId(options.relativePath);
  const physicalSessionProjectionId = createPhysicalSessionProjectionId(physicalSourceId);
  const evidenceOccurrences = createEvidenceOccurrences({
    imported,
    physicalSourceId,
    physicalSessionProjectionId,
    projectIdentityByOrigin: new Map(),
  });
  return {
    format: imported.format,
    evidenceOccurrences,
    denseOccurrences: evidenceOccurrences.filter(({ chunk }) => chunk.isDenseSearchable),
  };
}
