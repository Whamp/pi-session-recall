import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { open, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecOpen, type ZVecCollection, type ZVecStatus } from '@zvec/zvec';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { createInitialRecallPhysicalProjection } from './create-recall-session-projection-baseline.js';
import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallEligibilityThreshold,
  RecallIncrementalTransferOutcomeKind,
  RecallProjectionEncodingStatus,
  RecallSessionProjectionKind,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { materializeIncrementalRecallEligibleGraphView } from './materialize-incremental-recall-eligible-graph-view.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  createRecallGenerationManifest,
  readRecallGenerationManifest,
} from './recall-generation-manifest.js';
import {
  createRecallPhysicalSourceStoreMembership,
  parseRecallGenerationPhysicalProjectionArtifact,
  type RecallPhysicalSourceExpectedMembership,
} from './recall-generation-physical-projection.js';
import {
  createRecallGenerationComponentPaths,
  readRecallGenerationVectorValues,
  type RecallGenerationStoreContract,
} from './recall-generation-stores.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { acknowledgeCoveredRecallMarkers } from './recall-marker-spool.js';
import {
  decodeRecallSessionProjection,
  encodeRecallSessionProjection,
  mergeRecallMarkerCheckpoint,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallMarkerCheckpoint,
  type RecallSessionProjection,
} from './recall-session-projection.js';
import {
  createRecallEntryAnchorId,
  createRecallEvidenceOccurrenceId,
  createRecallLogicalSessionOccurrenceId,
  resolveRecallPhysicalSourceIdentity,
  type RecallPhysicalSourceIdentity,
} from './recall-source-identity.js';
import { createStoredRecallEmbedding } from './recall-stored-embedding.js';
import {
  readRecallSessionAppendDelta,
  type RecallSessionSourceRangeReader,
} from './read-recall-session-append-delta.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import { scheduleRecallWorkPlanEligibility } from './schedule-recall-work-plan-eligibility.js';
import {
  buildSessionConversationDocuments,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';
import { syncRecallDirectory } from './sync-recall-directory.js';
import {
  type ExactZvecDocumentEnumeration,
  visitExactZvecDocuments,
} from './visit-exact-zvec-documents.js';
import { serializeStoredConversationChunk } from './zvec-conversation-store.js';

/** Maximum evidence occurrences mutated in one target-generation write window. */
export const INCREMENTAL_RECALL_TARGET_EVIDENCE_BATCH_SIZE = 32;

interface RecallGenerationScalarRow {
  id: string;
  fields: Record<string, unknown>;
}

interface RecallGenerationDenseRow extends RecallGenerationScalarRow {
  vectors: { embedding: number[] };
}

interface PreparedTargetRecallTransfer {
  physicalSource: RecallPhysicalSourceIdentity;
  physicalProjection: PhysicalSessionProjection;
  logicalProjections: LogicalSessionProjection[];
  lexicalRows: RecallGenerationScalarRow[];
  denseRows: RecallGenerationDenseRow[];
  logicalProjectionRows: RecallGenerationScalarRow[];
  physicalProjectionRow: RecallGenerationScalarRow;
}

interface PhysicalSourceRecordIds {
  lexicalSource: string[];
  dense: string[];
  sessionProjection: string[];
}

interface IncrementalRecallTransferServices {
  generation: Readonly<RecallCoherentGenerationConfig> & Readonly<{ lockPath: string }>;
  chunkPolicy: RecallChunkPolicy;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  embeddingProvider: Pick<RecallEmbeddingProvider, 'embedDocuments'>;
  resolveProjectIdentity(sessionOrigin: string): Promise<ResolvedProjectIdentity | null>;
  readRange?: RecallSessionSourceRangeReader;
  incrementalTransferFault?: (
    stage:
      | 'before-recovery-record'
      | 'after-recovery-record'
      | 'after-lexical-source-write'
      | 'after-dense-write'
      | 'after-logical-projection-write'
      | 'after-physical-projection-write'
      | 'after-dense-delete'
      | 'after-lexical-source-delete'
      | 'after-projection-delete'
      | 'after-store-close'
      | 'after-reopened-verification'
      | 'after-recovery-clear'
      | 'after-marker-acknowledgement',
    context: Readonly<{
      generationId: string;
      physicalSourceIdentity: string;
      batchIndex: number;
      evidenceDocumentCount: number;
    }>,
  ) => void | Promise<void>;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
}

interface MarkerIncrementalRecallWorkPlanOptions extends IncrementalRecallTransferServices {
  workPlan: RecallMarkerReplayWorkPlan;
  confirmedPhysicalSourceDeletion?: never;
  physicalSessionProjectionUpdate?: never;
}

/** One policy-confirmed source removal routed through the sole incremental transfer seam. */
export interface ConfirmedPhysicalSourceDeletionRequest {
  targetGenerationId: string;
  physicalSourceIdentity: string;
}

interface ConfirmedDeletionIncrementalRecallWorkPlanOptions extends IncrementalRecallTransferServices {
  workPlan?: never;
  confirmedPhysicalSourceDeletion: ConfirmedPhysicalSourceDeletionRequest;
  physicalSessionProjectionUpdate?: never;
}

/** One policy update to the target generation's sole mutable physical-source account. */
export interface PhysicalSessionProjectionUpdateRequest {
  workPlan: RecallMarkerReplayWorkPlan;
  projection: PhysicalSessionProjection;
  /** False keeps deletion markers pending until confirmed store removal verifies. */
  acknowledgeMarkers?: boolean;
}

interface PhysicalSessionProjectionUpdateOptions extends IncrementalRecallTransferServices {
  workPlan?: never;
  confirmedPhysicalSourceDeletion?: never;
  physicalSessionProjectionUpdate: PhysicalSessionProjectionUpdateRequest;
}

/** Public service request selecting marker transfer, projection update, or confirmed deletion. */
export type IncrementalRecallWorkPlanRequest =
  | RecallMarkerReplayWorkPlan
  | Readonly<{ confirmedPhysicalSourceDeletion: ConfirmedPhysicalSourceDeletionRequest }>
  | Readonly<{ physicalSessionProjectionUpdate: PhysicalSessionProjectionUpdateRequest }>;

/** Target generation services plus exactly one supported physical-source transfer request. */
export type TransferIncrementalRecallWorkPlanOptions =
  | MarkerIncrementalRecallWorkPlanOptions
  | ConfirmedDeletionIncrementalRecallWorkPlanOptions
  | PhysicalSessionProjectionUpdateOptions;

/** Successful incremental transfer with the exact number of immutable documents committed. */
export interface CommittedIncrementalRecallWorkPlan {
  readonly kind: RecallIncrementalTransferOutcomeKind.COMMITTED;
  readonly committedDocumentCount: number;
}

/** Marker-backed work retained until one reconstructed quiet-period deadline. */
export interface DeferredIncrementalRecallWorkPlan {
  readonly kind: RecallIncrementalTransferOutcomeKind.DEFERRED;
  readonly threshold: RecallEligibilityThreshold;
  readonly readyAtEpochMilliseconds: number;
}

/** Durable outcome from one target-generation incremental transfer attempt. */
export type IncrementalRecallWorkPlanTransferOutcome =
  | CommittedIncrementalRecallWorkPlan
  | DeferredIncrementalRecallWorkPlan;

function calculateSha256(value: string | Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}

function assertSinglePhysicalSourceWorkPlan(workPlan: RecallMarkerReplayWorkPlan): void {
  const firstMarker = workPlan.workItems[0]?.marker;
  if (firstMarker === undefined) {
    throw new Error('Recall target incremental transfer requires at least one marker work item');
  }
  for (const { marker } of workPlan.workItems) {
    if (marker.physicalSessionPath !== firstMarker.physicalSessionPath) {
      throw new Error('Recall target incremental transfer work plan mixes physical sources');
    }
  }
}

function createDeferredTransfer(
  threshold: RecallEligibilityThreshold,
  readyAtEpochMilliseconds: number,
): DeferredIncrementalRecallWorkPlan {
  return {
    kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
    threshold,
    readyAtEpochMilliseconds,
  };
}

function targetGenerationIdForIncrementalTransfer(
  options: TransferIncrementalRecallWorkPlanOptions,
): string {
  if (options.confirmedPhysicalSourceDeletion !== undefined) {
    return options.confirmedPhysicalSourceDeletion.targetGenerationId;
  }
  if (options.physicalSessionProjectionUpdate !== undefined) {
    return options.physicalSessionProjectionUpdate.workPlan.targetGenerationId;
  }
  return options.workPlan.targetGenerationId;
}

async function assertConfiguredIncrementalTransferManifest(
  options: TransferIncrementalRecallWorkPlanOptions,
  generationId: string,
): Promise<void> {
  const generationDirectory = join(options.generation.generationRootDirectory, generationId);
  const manifestPath = createRecallGenerationComponentPaths(generationDirectory).manifestPath;
  const actualManifest = await readRecallGenerationManifest(manifestPath);
  const expectedManifest = createRecallGenerationManifest({
    generationId,
    embeddingProfileId: options.generation.embeddingProfileId,
    embeddingProfile: options.generation.embeddingProfile,
    projectLineages: options.generation.projectLineages,
    chunkPolicy: options.chunkPolicy,
  });
  assertRecallGenerationManifestCompatible(actualManifest.manifest, expectedManifest, manifestPath);
}

async function assertIncrementalTransferTargetRemainsActive(
  options: IncrementalRecallTransferServices,
  generationId: string,
): Promise<void> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.generation.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generation.generationRegistryPath),
  ]);
  if (
    pointer?.activeGenerationId !== generationId ||
    registry?.activeGenerationId !== generationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall target incremental transfer generation is no longer active');
  }
}

function fieldsMatch(
  actual: Readonly<Record<string, unknown>>,
  expected: Readonly<Record<string, unknown>>,
): boolean {
  return Object.entries(expected).every(
    ([name, expectedValue]) => JSON.stringify(actual[name]) === JSON.stringify(expectedValue),
  );
}

function assertCheckedStatuses(
  operation: string,
  recordIds: readonly string[],
  statuses: readonly ZVecStatus[],
): void {
  if (statuses.length !== recordIds.length) {
    throw new Error(
      `Recall target incremental ${operation} status mismatch: expected ${recordIds.length}, received ${statuses.length}`,
    );
  }
  for (const [index, status] of statuses.entries()) {
    if (!status.ok) {
      throw new Error(
        `Recall target incremental ${operation} failed for ${recordIds[index] ?? 'unknown record'}: ${status.message}`,
      );
    }
  }
}

function upsertMissingOrDamagedRows(
  collection: ZVecCollection,
  operation: string,
  rows: readonly (RecallGenerationScalarRow | RecallGenerationDenseRow)[],
): void {
  if (rows.length === 0) {
    return;
  }
  const outputFields = [...new Set(rows.flatMap(({ fields }) => Object.keys(fields)))];
  const includeVector = rows.some((row) => 'vectors' in row);
  const fetched = collection.fetchSync({
    ids: rows.map(({ id }) => id),
    outputFields,
    includeVector,
  });
  for (const row of rows) {
    const actual = fetched[row.id];
    if (
      row.fields.recordKind !== 'entry-anchor' ||
      actual === undefined ||
      !Array.isArray(actual.fields.evidenceOccurrenceIds) ||
      !Array.isArray(row.fields.evidenceOccurrenceIds)
    ) {
      continue;
    }
    const evidenceOccurrenceIds = [
      ...new Set([
        ...actual.fields.evidenceOccurrenceIds.filter(
          (value): value is string => typeof value === 'string',
        ),
        ...row.fields.evidenceOccurrenceIds.filter(
          (value): value is string => typeof value === 'string',
        ),
      ]),
    ].toSorted();
    row.fields.evidenceOccurrenceIds = evidenceOccurrenceIds;
    if (typeof row.fields.recordJson === 'string') {
      let anchorRecord: unknown;
      try {
        anchorRecord = JSON.parse(row.fields.recordJson);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Recall target incremental entry anchor JSON invalid for ${row.id}: ${message}`,
          { cause: error },
        );
      }
      if (typeof anchorRecord !== 'object' || anchorRecord === null) {
        throw new Error(`Recall target incremental entry anchor record invalid: ${row.id}`);
      }
      row.fields.recordJson = JSON.stringify({ ...anchorRecord, evidenceOccurrenceIds });
    }
  }
  const requiringWrite = rows.filter((row) => {
    const actual = fetched[row.id];
    if (actual === undefined || !fieldsMatch(actual.fields, row.fields)) {
      return true;
    }
    if (!('vectors' in row)) {
      return false;
    }
    return (
      calculateSha256(
        Buffer.from(
          new Float32Array(readRecallGenerationVectorValues(actual.vectors.embedding)).buffer,
        ),
      ) !== row.fields.vectorChecksum
    );
  });
  if (requiringWrite.length === 0) {
    return;
  }
  const damagedIds = requiringWrite
    .filter(({ id }) => fetched[id] !== undefined)
    .map(({ id }) => id);
  if (damagedIds.length > 0) {
    assertCheckedStatuses(operation, damagedIds, collection.deleteSync(damagedIds));
  }
  assertCheckedStatuses(
    operation,
    requiringWrite.map(({ id }) => id),
    collection.upsertSync(requiringWrite),
  );
}

function verifyRows(
  collection: ZVecCollection,
  responsibility: string,
  rows: readonly (RecallGenerationScalarRow | RecallGenerationDenseRow)[],
): void {
  if (rows.length === 0) {
    return;
  }
  const outputFields = [...new Set(rows.flatMap(({ fields }) => Object.keys(fields)))];
  const includeVector = rows.some((row) => 'vectors' in row);
  const fetched = collection.fetchSync({
    ids: rows.map(({ id }) => id),
    outputFields,
    includeVector,
  });
  for (const row of rows) {
    const actual = fetched[row.id];
    if (actual === undefined || !fieldsMatch(actual.fields, row.fields)) {
      throw new Error(`Recall target incremental ${responsibility} row mismatch: ${row.id}`);
    }
    if ('vectors' in row) {
      const checksum = calculateSha256(
        Buffer.from(
          new Float32Array(readRecallGenerationVectorValues(actual.vectors.embedding)).buffer,
        ),
      );
      if (checksum !== row.fields.vectorChecksum) {
        throw new Error(`Recall target incremental dense vector mismatch: ${row.id}`);
      }
    }
  }
}

function parseIngestionProjection(
  projectionJson: unknown,
  generationId: string,
): RecallSessionProjection {
  if (typeof projectionJson !== 'string') {
    throw new Error('Recall target incremental session projection JSON missing');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(projectionJson);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall target incremental session projection JSON invalid: ${message}`, {
      cause: error,
    });
  }
  if (typeof parsed !== 'object' || parsed === null || !('ingestionProjectionPayload' in parsed)) {
    throw new Error('Recall target incremental ingestion projection missing');
  }
  return decodeRecallSessionProjection(parsed.ingestionProjectionPayload, {
    expectedGenerationId: generationId,
  });
}

function readCurrentTargetProjections(
  options: MarkerIncrementalRecallWorkPlanOptions,
  physicalSourceIdentity: string,
): {
  physicalProjection?: PhysicalSessionProjection;
  logicalProjections: LogicalSessionProjection[];
} {
  const paths = createRecallGenerationComponentPaths(
    join(options.generation.generationRootDirectory, options.workPlan.targetGenerationId),
  );
  const collection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  try {
    const physicalRowId = `projection_${physicalSourceIdentity}`;
    const fetchedPhysical = collection.fetchSync({
      ids: [physicalRowId],
      outputFields: ['projectionJson'],
      includeVector: false,
    })[physicalRowId];
    if (fetchedPhysical === undefined) {
      return { logicalProjections: [] };
    }
    const physicalProjection = parseIngestionProjection(
      fetchedPhysical.fields.projectionJson,
      options.workPlan.targetGenerationId,
    );
    if (physicalProjection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
      throw new Error('Recall target incremental physical projection kind mismatch');
    }
    const logicalRowIds = physicalProjection.logicalSessionIds.map((logicalSessionId) => {
      const logicalProjection = createRecallLogicalSessionOccurrenceId(
        physicalSourceIdentity,
        Number(logicalSessionId.split('@').at(-1)),
      );
      return `projection_${logicalProjection}`;
    });
    const fetchedLogical =
      logicalRowIds.length === 0
        ? {}
        : collection.fetchSync({
            ids: logicalRowIds,
            outputFields: ['projectionJson'],
            includeVector: false,
          });
    const logicalProjections = logicalRowIds.map((rowId) => {
      const row = fetchedLogical[rowId];
      if (row === undefined) {
        throw new Error(`Recall target incremental logical projection missing: ${rowId}`);
      }
      const projection = parseIngestionProjection(
        row.fields.projectionJson,
        options.workPlan.targetGenerationId,
      );
      if (projection.projectionKind !== RecallSessionProjectionKind.LOGICAL_SESSION) {
        throw new Error(`Recall target incremental logical projection kind mismatch: ${rowId}`);
      }
      return projection;
    });
    return { physicalProjection, logicalProjections };
  } finally {
    collection.closeSync();
  }
}

function spansForLogicalProjection(
  spans: readonly RecallEligibleSourceSpan[],
  projection: LogicalSessionProjection,
): RecallEligibleSourceSpan[] {
  const boundaries = new Set(
    projection.entryDescriptors.map(({ startByte, endByte }) => `${startByte}:${endByte}`),
  );
  return spans.filter(({ startByte, endByte }) => boundaries.has(`${startByte}:${endByte}`));
}

function createOccurrenceId(
  physicalSourceIdentity: string,
  logicalSessionOccurrenceId: string,
  document: SessionConversationChunk,
): string {
  return createRecallEvidenceOccurrenceId({
    physicalSourceIdentity,
    logicalSessionOccurrenceId,
    entryId: document.entryId.value,
    evidencePart: document.evidencePart,
    sourceLineStart: document.sourceLineStart,
    sourceLineEnd: document.sourceLineEnd,
    sourceBlockStart: document.sourceBlockStart,
    sourceBlockEnd: document.sourceBlockEnd,
    characterStart: document.characterStart,
    characterEnd: document.characterEnd,
    tokenStart: document.tokenStart,
    tokenEnd: document.tokenEnd,
    textRunIndex: document.textRunIndex,
    chunkIndex: document.chunkIndex,
  });
}

function createBranchLeafIdsByEntryId(projection: LogicalSessionProjection): Map<string, string[]> {
  const childIdsByEntryId = new Map<string, string[]>();
  for (const descriptor of projection.entryDescriptors) {
    if (descriptor.parentEntryId !== null) {
      const children = childIdsByEntryId.get(descriptor.parentEntryId) ?? [];
      children.push(descriptor.entryId);
      childIdsByEntryId.set(descriptor.parentEntryId, children);
    }
  }
  const leafIds = projection.entryDescriptors
    .map(({ entryId }) => entryId)
    .filter((entryId) => (childIdsByEntryId.get(entryId) ?? []).length === 0);
  const descriptorsById = new Map(
    projection.entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  const result = new Map<string, string[]>();
  for (const descriptor of projection.entryDescriptors) {
    result.set(
      descriptor.entryId,
      leafIds.filter((leafId) => {
        let current = descriptorsById.get(leafId);
        while (current !== undefined) {
          if (current.entryId === descriptor.entryId) {
            return true;
          }
          current =
            current.parentEntryId === null ? undefined : descriptorsById.get(current.parentEntryId);
        }
        return false;
      }),
    );
  }
  return result;
}

function createCommonLexicalFields(options: {
  generationId: string;
  physicalSource: RecallPhysicalSourceIdentity;
  logicalSessionOccurrenceId: string;
  rawSessionId: string;
  headerSourceLine: number;
  entryAnchorId: string;
  descriptor: LogicalSessionProjection['entryDescriptors'][number];
  childEntryIds: readonly string[];
  branchPathLeafIds: readonly string[];
  evidenceOccurrenceIds: readonly string[];
}): Record<string, unknown> {
  return {
    schemaVersion: 1,
    generationId: options.generationId,
    physicalSourceIdentity: options.physicalSource.physicalSourceIdentity,
    sessionsRootRelativePath: options.physicalSource.sessionsRootRelativePath,
    logicalSessionOccurrenceId: options.logicalSessionOccurrenceId,
    rawSessionId: options.rawSessionId,
    headerSourceLine: options.headerSourceLine,
    entryAnchorId: options.entryAnchorId,
    entryId: options.descriptor.entryId,
    parentEntryId: options.descriptor.parentEntryId ?? '',
    childEntryIds: [...options.childEntryIds],
    branchPathLeafIds: [...options.branchPathLeafIds],
    evidenceOccurrenceIds: [...options.evidenceOccurrenceIds],
    sourceOrder: options.descriptor.sourceLine,
    entryType: options.descriptor.entryType,
    timestamp: options.descriptor.timestamp,
    entryStartByte: options.descriptor.startByte,
    entryEndByte: options.descriptor.endByte,
  };
}

async function materializePreparedTargetTransfer(
  options: MarkerIncrementalRecallWorkPlanOptions,
  physicalSource: RecallPhysicalSourceIdentity,
  physicalProjection: PhysicalSessionProjection,
  logicalProjections: LogicalSessionProjection[],
  currentRecordIds: PhysicalSourceRecordIds,
  eligibleSessions: readonly Readonly<{
    graphView: Awaited<ReturnType<typeof materializeIncrementalRecallEligibleGraphView>>;
    logicalProjection: LogicalSessionProjection;
    newlyEligibleSpans: readonly RecallEligibleSourceSpan[];
  }>[],
  tokenizer: ConversationTextTokenizer,
): Promise<PreparedTargetRecallTransfer> {
  const generationId = options.workPlan.targetGenerationId;
  const lexicalRows: RecallGenerationScalarRow[] = [];
  const denseInputs: Array<{
    row: RecallGenerationScalarRow;
    embeddingInput: string;
  }> = [];
  const logicalProjectionRows: RecallGenerationScalarRow[] = [];
  for (const eligibleSession of eligibleSessions) {
    const projection = eligibleSession.logicalProjection;
    const logicalSessionOccurrenceId = createRecallLogicalSessionOccurrenceId(
      physicalSource.physicalSourceIdentity,
      projection.headerDescriptor.sourceLine,
    );
    const documents = buildSessionConversationDocuments(
      eligibleSession.graphView.graph,
      new Set(projection.eligibleContributorEntryIds),
      {
        sessionPath: eligibleSession.graphView.physicalPath,
        logicalSessionIdentity: projection.logicalSessionId,
        physicalSessionProjectionId: physicalSource.physicalSourceIdentity,
        newlyEligibleContributorEntryIds: new Set(
          eligibleSession.newlyEligibleSpans.flatMap(
            ({ contributorEntryIds }) => contributorEntryIds,
          ),
        ),
        tokenizer,
        maxTokens: options.chunkPolicy.maxTokens,
        overlapTokens: options.chunkPolicy.overlapTokens,
      },
    );
    const projectAttribution = await options.resolveProjectIdentity(
      projection.headerDescriptor.cwd,
    );
    const occurrenceIdByDocumentId = new Map(
      documents.map((document) => [
        document.id,
        createOccurrenceId(
          physicalSource.physicalSourceIdentity,
          logicalSessionOccurrenceId,
          document,
        ),
      ]),
    );
    const occurrenceIdsByEntryId = new Map<string, string[]>();
    for (const document of documents) {
      const occurrenceId = occurrenceIdByDocumentId.get(document.id);
      if (occurrenceId !== undefined) {
        const ids = occurrenceIdsByEntryId.get(document.entryId.value) ?? [];
        ids.push(occurrenceId);
        occurrenceIdsByEntryId.set(document.entryId.value, ids);
      }
    }
    const childIdsByEntryId = new Map<string, string[]>();
    for (const descriptor of projection.entryDescriptors) {
      if (descriptor.parentEntryId !== null) {
        const children = childIdsByEntryId.get(descriptor.parentEntryId) ?? [];
        children.push(descriptor.entryId);
        childIdsByEntryId.set(descriptor.parentEntryId, children);
      }
    }
    const branchLeafIdsByEntryId = createBranchLeafIdsByEntryId(projection);
    const entryAnchorIds: string[] = [];
    for (const descriptor of projection.entryDescriptors) {
      const entryAnchorId = createRecallEntryAnchorId({
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId,
        entryId: descriptor.entryId,
        sourceLine: descriptor.sourceLine,
        startByte: descriptor.startByte,
        endByte: descriptor.endByte,
      });
      entryAnchorIds.push(entryAnchorId);
      const common = createCommonLexicalFields({
        generationId,
        physicalSource,
        logicalSessionOccurrenceId,
        rawSessionId: projection.rawSessionId ?? eligibleSession.graphView.logicalSessionId,
        headerSourceLine: projection.headerDescriptor.sourceLine,
        entryAnchorId,
        descriptor,
        childEntryIds: childIdsByEntryId.get(descriptor.entryId) ?? [],
        branchPathLeafIds: branchLeafIdsByEntryId.get(descriptor.entryId) ?? [],
        evidenceOccurrenceIds: occurrenceIdsByEntryId.get(descriptor.entryId) ?? [],
      });
      lexicalRows.push({
        id: entryAnchorId,
        fields: {
          ...common,
          recordKind: 'entry-anchor',
          evidenceOccurrenceId: '',
          documentKind: '',
          evidenceKind: '',
          evidencePart: '',
          isDenseSearchable: false,
          evidenceChecksum: '',
          projectIdentity: projectAttribution?.projectIdentity ?? '',
          projectIdentityDigest: projectAttribution
            ? calculateSha256(projectAttribution.projectIdentity)
            : '',
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
            rawSessionId: projection.rawSessionId ?? eligibleSession.graphView.logicalSessionId,
            projectAttribution,
            descriptor,
            childEntryIds: childIdsByEntryId.get(descriptor.entryId) ?? [],
            branchPathLeafIds: branchLeafIdsByEntryId.get(descriptor.entryId) ?? [],
            evidenceOccurrenceIds: occurrenceIdsByEntryId.get(descriptor.entryId) ?? [],
          }),
        },
      });
    }
    for (const document of documents) {
      const descriptor = projection.entryDescriptors.find(
        ({ entryId }) => entryId === document.entryId.value,
      );
      const occurrenceId = occurrenceIdByDocumentId.get(document.id);
      if (descriptor === undefined || occurrenceId === undefined) {
        throw new Error(
          `Recall target incremental evidence source geometry missing: ${document.id}`,
        );
      }
      const entryAnchorId = createRecallEntryAnchorId({
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId,
        entryId: descriptor.entryId,
        sourceLine: descriptor.sourceLine,
        startByte: descriptor.startByte,
        endByte: descriptor.endByte,
      });
      const common = createCommonLexicalFields({
        generationId,
        physicalSource,
        logicalSessionOccurrenceId,
        rawSessionId: projection.rawSessionId ?? eligibleSession.graphView.logicalSessionId,
        headerSourceLine: projection.headerDescriptor.sourceLine,
        entryAnchorId,
        descriptor,
        childEntryIds: childIdsByEntryId.get(descriptor.entryId) ?? [],
        branchPathLeafIds: branchLeafIdsByEntryId.get(descriptor.entryId) ?? [],
        evidenceOccurrenceIds: occurrenceIdsByEntryId.get(descriptor.entryId) ?? [],
      });
      const targetDocument: SessionConversationChunk = {
        ...document,
        id: occurrenceId,
        physicalSessionProjectionId: physicalSource.physicalSourceIdentity,
        projectAttribution,
        siblingIds: document.siblingIds.map((id) => occurrenceIdByDocumentId.get(id) ?? id),
        previousSiblingId:
          document.previousSiblingId === null
            ? null
            : (occurrenceIdByDocumentId.get(document.previousSiblingId) ??
              document.previousSiblingId),
        nextSiblingId:
          document.nextSiblingId === null
            ? null
            : (occurrenceIdByDocumentId.get(document.nextSiblingId) ?? document.nextSiblingId),
      };
      const evidence = {
        evidenceOccurrenceId: occurrenceId,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        sessionsRootRelativePath: physicalSource.sessionsRootRelativePath,
        logicalSessionOccurrenceId,
        rawSessionId: projection.rawSessionId ?? eligibleSession.graphView.logicalSessionId,
        entryAnchorId,
        entryId: document.entryId.value,
        parentEntryId: document.parentEntryId?.value ?? null,
        evidenceKind: document.evidenceKind,
        evidencePart: document.evidencePart,
        isDenseSearchable: document.isDenseSearchable,
        projectIdentity: projectAttribution?.projectIdentity ?? '',
        sourceLineStart: document.sourceLineStart,
        sourceLineEnd: document.sourceLineEnd,
        content: document.content,
      };
      const row: RecallGenerationScalarRow = {
        id: occurrenceId,
        fields: {
          ...common,
          recordKind: 'evidence',
          evidenceOccurrenceId: occurrenceId,
          documentKind: document.documentKind,
          evidenceKind: document.evidenceKind,
          evidencePart: document.evidencePart,
          isDenseSearchable: document.isDenseSearchable,
          evidenceChecksum: document.checksum,
          projectIdentity: evidence.projectIdentity,
          projectIdentityDigest: evidence.projectIdentity
            ? calculateSha256(evidence.projectIdentity)
            : '',
          sourceLineStart: document.sourceLineStart,
          sourceLineEnd: document.sourceLineEnd,
          sourceBlockStart: document.sourceBlockStart ?? -1,
          sourceBlockEnd: document.sourceBlockEnd ?? -1,
          characterStart: document.characterStart,
          characterEnd: document.characterEnd,
          tokenStart: document.tokenStart,
          tokenEnd: document.tokenEnd,
          textRunIndex: document.textRunIndex,
          chunkIndex: document.chunkIndex,
          content: document.content,
          identifierContent: document.content,
          recordJson: JSON.stringify({
            evidence,
            searchDocument: serializeStoredConversationChunk(targetDocument),
          }),
        },
      };
      lexicalRows.push(row);
      if (document.isDenseSearchable) {
        denseInputs.push({ row, embeddingInput: document.content });
      }
    }
    const encodedLogical = encodeRecallSessionProjection(projection);
    if (encodedLogical.status !== RecallProjectionEncodingStatus.ENCODED) {
      throw new Error(
        `Recall target incremental logical projection exceeds the bounded payload: ${projection.logicalSessionId}`,
      );
    }
    logicalProjectionRows.push({
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
          rawSessionId: projection.rawSessionId ?? eligibleSession.graphView.logicalSessionId,
          headerSourceLine: projection.headerDescriptor.sourceLine,
          projectAttribution,
          entryAnchorIds,
          ingestionProjectionPayload: encodedLogical.payload,
        }),
      },
    });
  }
  const denseNativeVectors = await options.embeddingProvider.embedDocuments(
    denseInputs.map(({ embeddingInput }) => embeddingInput),
    options.signal,
  );
  if (denseNativeVectors.length !== denseInputs.length) {
    throw new Error(
      `Recall target incremental document embedding count mismatch: expected ${denseInputs.length}, received ${denseNativeVectors.length}`,
    );
  }
  const storedDimensions =
    options.generation.embeddingProfile.storedDimensions ??
    options.generation.embeddingProfile.identity.dimensions;
  const denseRows = denseInputs.map(({ row, embeddingInput }, index): RecallGenerationDenseRow => {
    const nativeVector = denseNativeVectors[index];
    if (nativeVector === undefined) {
      throw new Error(`Recall target incremental document embedding missing: ${row.id}`);
    }
    const embedding = createStoredRecallEmbedding(nativeVector, {
      nativeDimensions: options.generation.embeddingProfile.identity.dimensions,
      storedDimensions,
      source: `generation ${generationId}:${row.id}`,
    });
    return {
      id: row.id,
      fields: {
        schemaVersion: 1,
        generationId,
        evidenceOccurrenceId: row.id,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
        logicalSessionOccurrenceId: row.fields.logicalSessionOccurrenceId,
        embeddingProfileId: options.generation.embeddingProfileId,
        storedDimensions,
        evidenceChecksum: row.fields.evidenceChecksum,
        embeddingInputChecksum: calculateSha256(
          `${options.generation.embeddingProfile.documentInputPrefix}${embeddingInput}`,
        ),
        vectorChecksum: calculateSha256(Buffer.from(new Float32Array(embedding).buffer)),
        projectIdentity: row.fields.projectIdentity,
        projectIdentityDigest: row.fields.projectIdentityDigest,
      },
      vectors: { embedding },
    };
  });
  const encodedPhysical = encodeRecallSessionProjection(physicalProjection);
  if (encodedPhysical.status !== RecallProjectionEncodingStatus.ENCODED) {
    throw new Error('Recall target incremental physical projection exceeds the bounded payload');
  }
  const physicalProjectionRow: RecallGenerationScalarRow = {
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
        sourceByteSize: physicalProjection.appendCursorBytes,
        logicalSessionOccurrenceIds: logicalProjections.map((projection) =>
          createRecallLogicalSessionOccurrenceId(
            physicalSource.physicalSourceIdentity,
            projection.headerDescriptor.sourceLine,
          ),
        ),
        expectedMembership: createPhysicalSourceExpectedMembership({
          lexicalSource: [
            ...new Set([...currentRecordIds.lexicalSource, ...lexicalRows.map(({ id }) => id)]),
          ],
          dense: [...new Set([...currentRecordIds.dense, ...denseRows.map(({ id }) => id)])],
          sessionProjection: [
            ...new Set([
              ...currentRecordIds.sessionProjection,
              ...logicalProjectionRows.map(({ id }) => id),
              `projection_${physicalSource.physicalSourceIdentity}`,
            ]),
          ],
        }),
        ingestionProjectionPayload: encodedPhysical.payload,
      }),
    },
  };
  return {
    physicalSource,
    physicalProjection,
    logicalProjections,
    lexicalRows,
    denseRows,
    logicalProjectionRows,
    physicalProjectionRow,
  };
}

async function writeRecoveryRecord(
  recoveryRecordPath: string,
  recovery: Readonly<Record<string, unknown>>,
): Promise<void> {
  const expected = `${JSON.stringify(recovery, null, 2)}\n`;
  if (existsSync(recoveryRecordPath)) {
    const actual = await readFile(recoveryRecordPath, 'utf8');
    if (actual !== expected) {
      throw new Error(
        'Recall target incremental recovery record does not match the replayed batch',
      );
    }
    return;
  }
  await writeFile(recoveryRecordPath, expected, { encoding: 'utf8', flag: 'wx' });
  const handle = await open(recoveryRecordPath, 'r');
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
  await syncRecallDirectory(join(recoveryRecordPath, '..'));
}

async function invokeIncrementalTransferFault(
  options: MarkerIncrementalRecallWorkPlanOptions,
  stage: Parameters<NonNullable<IncrementalRecallTransferServices['incrementalTransferFault']>>[0],
  prepared: PreparedTargetRecallTransfer,
  batchIndex: number,
  evidenceDocumentCount: number,
): Promise<void> {
  await options.incrementalTransferFault?.(stage, {
    generationId: options.workPlan.targetGenerationId,
    physicalSourceIdentity: prepared.physicalSource.physicalSourceIdentity,
    batchIndex,
    evidenceDocumentCount,
  });
}

function parseIncrementalRecoveryScalarRows(
  value: unknown,
  fieldName: string,
): RecallGenerationScalarRow[] {
  if (!Array.isArray(value)) {
    throw new Error(`Recall target incremental recovery ${fieldName} invalid`);
  }
  return value.map((candidate) => {
    if (
      !isUnknownRecord(candidate) ||
      typeof candidate.id !== 'string' ||
      !isUnknownRecord(candidate.fields)
    ) {
      throw new Error(`Recall target incremental recovery ${fieldName} invalid`);
    }
    return { id: candidate.id, fields: { ...candidate.fields } };
  });
}

function parseIncrementalRecoveryDenseRows(value: unknown): RecallGenerationDenseRow[] {
  return parseIncrementalRecoveryScalarRows(value, 'dense rows').map((row, index) => {
    if (!Array.isArray(value)) {
      throw new Error('Recall target incremental recovery dense rows invalid');
    }
    const candidate: unknown = value[index];
    if (
      !isUnknownRecord(candidate) ||
      !isUnknownRecord(candidate.vectors) ||
      !Array.isArray(candidate.vectors.embedding)
    ) {
      throw new Error('Recall target incremental recovery dense vectors invalid');
    }
    const embedding: number[] = [];
    for (const component of candidate.vectors.embedding) {
      if (typeof component !== 'number' || !Number.isFinite(component)) {
        throw new Error('Recall target incremental recovery dense vector invalid');
      }
      embedding.push(component);
    }
    return { ...row, vectors: { embedding } };
  });
}

async function recoverPendingIncrementalTransfer(
  options: MarkerIncrementalRecallWorkPlanOptions,
  physicalSourceIdentity: string,
): Promise<CommittedIncrementalRecallWorkPlan | null> {
  const generationId = options.workPlan.targetGenerationId;
  const generationDirectory = join(options.generation.generationRootDirectory, generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  if (!existsSync(paths.recoveryRecordPath)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(paths.recoveryRecordPath, 'utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall target incremental recovery record invalid: ${message}`, {
      cause: error,
    });
  }
  if (!isUnknownRecord(parsed) || parsed.operation !== 'incremental-physical-source') {
    return null;
  }
  if (
    parsed.generationId !== generationId ||
    parsed.physicalSourceIdentity !== physicalSourceIdentity
  ) {
    throw new Error('Recall target incremental recovery record identity mismatch');
  }
  const lexicalRows = parseIncrementalRecoveryScalarRows(parsed.evidenceRows, 'lexical rows');
  const denseRows = parseIncrementalRecoveryDenseRows(parsed.denseRows);
  const projectionRows = parseIncrementalRecoveryScalarRows(
    parsed.projectionRows,
    'projection rows',
  );
  await coordinateRecallWriteWindow(
    {
      lockPath: options.generation.lockPath,
      allowRecovery: true,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      await assertIncrementalTransferTargetRemainsActive(options, generationId);
      const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
      const dense = ZVecOpen(paths.denseStorePath);
      const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
      let operationError: unknown;
      try {
        upsertMissingOrDamagedRows(lexicalSource, 'recovery lexical/source write', lexicalRows);
        upsertMissingOrDamagedRows(dense, 'recovery dense write', denseRows);
        upsertMissingOrDamagedRows(sessionProjection, 'recovery projection write', projectionRows);
      } catch (error) {
        operationError = error;
      }
      const closeErrors: unknown[] = [];
      for (const store of [sessionProjection, dense, lexicalSource]) {
        try {
          store.closeSync();
        } catch (error) {
          closeErrors.push(error);
        }
      }
      if (operationError !== undefined || closeErrors.length > 0) {
        writeWindow.retainRecoveryRequired();
        throw new AggregateError(
          [operationError, ...closeErrors].filter((error) => error !== undefined),
          'Recall target incremental recovery write or close failed',
        );
      }
      const reopenedLexical = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
      const reopenedDense = ZVecOpen(paths.denseStorePath, { readOnly: true });
      const reopenedProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
      try {
        verifyRows(reopenedLexical, 'recovery lexical/source checkpoint', lexicalRows);
        verifyRows(reopenedDense, 'recovery dense checkpoint', denseRows);
        verifyRows(reopenedProjection, 'recovery projection checkpoint', projectionRows);
        if (
          projectionRows.some(
            ({ fields }) => fields.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
          )
        ) {
          verifyPhysicalSourceExpectedMembership(
            reopenedLexical,
            reopenedDense,
            reopenedProjection,
            generationId,
            physicalSourceIdentity,
          );
        }
      } finally {
        reopenedProjection.closeSync();
        reopenedDense.closeSync();
        reopenedLexical.closeSync();
      }
      await rm(paths.recoveryRecordPath);
      await syncRecallDirectory(generationDirectory);
      writeWindow.attestRecoveryCompleted();
    },
  );
  const physicalProjectionRow = projectionRows.find(
    ({ fields }) => fields.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
  );
  if (physicalProjectionRow === undefined) {
    return null;
  }
  const projection = parseIngestionProjection(
    physicalProjectionRow.fields.projectionJson,
    generationId,
  );
  if (projection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
    throw new Error('Recall target incremental recovered physical projection kind mismatch');
  }
  await acknowledgeCoveredRecallMarkers(options.workPlan, projection.markerCheckpoint);
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: lexicalRows.filter(({ fields }) => fields.recordKind === 'evidence')
      .length,
  };
}

async function commitPreparedTargetTransfer(
  options: MarkerIncrementalRecallWorkPlanOptions,
  prepared: PreparedTargetRecallTransfer,
): Promise<RecallMarkerCheckpoint> {
  const generationId = options.workPlan.targetGenerationId;
  const generationDirectory = join(options.generation.generationRootDirectory, generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  const evidenceRows = prepared.lexicalRows.filter(
    ({ fields }) => fields.recordKind === 'evidence',
  );
  const anchorRows = prepared.lexicalRows.filter(
    ({ fields }) => fields.recordKind === 'entry-anchor',
  );
  const batches: RecallGenerationScalarRow[][] = [];
  for (
    let offset = 0;
    offset < evidenceRows.length;
    offset += INCREMENTAL_RECALL_TARGET_EVIDENCE_BATCH_SIZE
  ) {
    batches.push(
      evidenceRows.slice(offset, offset + INCREMENTAL_RECALL_TARGET_EVIDENCE_BATCH_SIZE),
    );
  }
  if (batches.length === 0) {
    batches.push([]);
  }
  for (const [batchIndex, evidenceBatch] of batches.entries()) {
    const finalWindow = batchIndex === batches.length - 1;
    const evidenceIds = new Set(evidenceBatch.map(({ id }) => id));
    const denseBatch = prepared.denseRows.filter(({ id }) => evidenceIds.has(id));
    const lexicalBatch = batchIndex === 0 ? [...anchorRows, ...evidenceBatch] : evidenceBatch;
    const projectionRows = finalWindow
      ? [...prepared.logicalProjectionRows, prepared.physicalProjectionRow]
      : [];
    const recovery = {
      version: 1,
      generationId,
      operation: 'incremental-physical-source',
      physicalSourceIdentity: prepared.physicalSource.physicalSourceIdentity,
      batchIndex,
      evidenceRows: lexicalBatch,
      denseRows: denseBatch,
      projectionRows,
      coveredMarkerIds: options.workPlan.sourceMarkerIds,
    };
    await coordinateRecallWriteWindow(
      {
        lockPath: options.generation.lockPath,
        allowRecovery: true,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      async (writeWindow) => {
        await assertIncrementalTransferTargetRemainsActive(options, generationId);
        await invokeIncrementalTransferFault(
          options,
          'before-recovery-record',
          prepared,
          batchIndex,
          evidenceBatch.length,
        );
        await writeRecoveryRecord(paths.recoveryRecordPath, recovery);
        await invokeIncrementalTransferFault(
          options,
          'after-recovery-record',
          prepared,
          batchIndex,
          evidenceBatch.length,
        );
        let lexicalSource: ZVecCollection | undefined;
        let dense: ZVecCollection | undefined;
        let sessionProjection: ZVecCollection | undefined;
        let operationError: unknown;
        try {
          lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
          dense = ZVecOpen(paths.denseStorePath);
          sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
          upsertMissingOrDamagedRows(lexicalSource, 'lexical/source write', lexicalBatch);
          await invokeIncrementalTransferFault(
            options,
            'after-lexical-source-write',
            prepared,
            batchIndex,
            evidenceBatch.length,
          );
          upsertMissingOrDamagedRows(dense, 'dense write', denseBatch);
          await invokeIncrementalTransferFault(
            options,
            'after-dense-write',
            prepared,
            batchIndex,
            evidenceBatch.length,
          );
          if (finalWindow) {
            upsertMissingOrDamagedRows(
              sessionProjection,
              'logical projection write',
              prepared.logicalProjectionRows,
            );
            await invokeIncrementalTransferFault(
              options,
              'after-logical-projection-write',
              prepared,
              batchIndex,
              evidenceBatch.length,
            );
            upsertMissingOrDamagedRows(sessionProjection, 'physical projection write', [
              prepared.physicalProjectionRow,
            ]);
            await invokeIncrementalTransferFault(
              options,
              'after-physical-projection-write',
              prepared,
              batchIndex,
              evidenceBatch.length,
            );
          }
        } catch (error) {
          operationError = error;
        }
        const closeErrors: unknown[] = [];
        for (const store of [sessionProjection, dense, lexicalSource]) {
          try {
            store?.closeSync();
          } catch (error) {
            closeErrors.push(error);
          }
        }
        if (operationError === undefined && closeErrors.length === 0) {
          try {
            await invokeIncrementalTransferFault(
              options,
              'after-store-close',
              prepared,
              batchIndex,
              evidenceBatch.length,
            );
          } catch (error) {
            operationError = error;
          }
        }
        if (operationError !== undefined || closeErrors.length > 0) {
          writeWindow.retainRecoveryRequired();
          throw new AggregateError(
            [operationError, ...closeErrors].filter((error) => error !== undefined),
            'Recall target incremental write or close failed',
          );
        }
        const reopenedLexical = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
        const reopenedDense = ZVecOpen(paths.denseStorePath, { readOnly: true });
        const reopenedProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
        try {
          verifyRows(reopenedLexical, 'lexical/source checkpoint', lexicalBatch);
          verifyRows(reopenedDense, 'dense checkpoint', denseBatch);
          verifyRows(reopenedProjection, 'projection checkpoint', projectionRows);
          if (finalWindow) {
            verifyPhysicalSourceExpectedMembership(
              reopenedLexical,
              reopenedDense,
              reopenedProjection,
              generationId,
              prepared.physicalSource.physicalSourceIdentity,
            );
          }
          await invokeIncrementalTransferFault(
            options,
            'after-reopened-verification',
            prepared,
            batchIndex,
            evidenceBatch.length,
          );
        } finally {
          reopenedProjection.closeSync();
          reopenedDense.closeSync();
          reopenedLexical.closeSync();
        }
        await rm(paths.recoveryRecordPath);
        await syncRecallDirectory(generationDirectory);
        writeWindow.attestRecoveryCompleted();
        await invokeIncrementalTransferFault(
          options,
          'after-recovery-clear',
          prepared,
          batchIndex,
          evidenceBatch.length,
        );
      },
    );
  }
  const projectionStore = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  try {
    const row = projectionStore.fetchSync({
      ids: [prepared.physicalProjectionRow.id],
      outputFields: ['projectionJson'],
      includeVector: false,
    })[prepared.physicalProjectionRow.id];
    if (row === undefined) {
      throw new Error(
        'Recall target incremental physical projection missing after reopened verification',
      );
    }
    const projection = parseIngestionProjection(row.fields.projectionJson, generationId);
    if (projection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
      throw new Error(
        'Recall target incremental physical projection kind mismatch after verification',
      );
    }
    return projection.markerCheckpoint;
  } finally {
    projectionStore.closeSync();
  }
}

function createPhysicalSourceRecordEnumerations(
  physicalSourceIdentity: string,
  responsibility: RecallGenerationStoreContract['responsibility'],
): readonly ExactZvecDocumentEnumeration[] {
  const sourceFilter = `physicalSourceIdentity = '${physicalSourceIdentity}'`;
  switch (responsibility) {
    case 'lexical-source':
      return [
        {
          filter: `(${sourceFilter}) AND (recordKind = 'entry-anchor')`,
          uniquePartitionField: 'entryAnchorId',
          outputFields: [],
        },
        {
          filter: `(${sourceFilter}) AND (recordKind = 'evidence')`,
          uniquePartitionField: 'evidenceOccurrenceId',
          outputFields: [],
        },
      ];
    case 'dense-evidence':
      return [
        {
          filter: sourceFilter,
          uniquePartitionField: 'evidenceOccurrenceId',
          outputFields: [],
        },
      ];
    case 'session-projection':
      return [
        {
          filter: `(${sourceFilter}) AND (projectionKind = '${RecallSessionProjectionKind.PHYSICAL_SESSION}')`,
          uniquePartitionField: 'physicalSourceIdentity',
          outputFields: [],
        },
        {
          filter: `(${sourceFilter}) AND (projectionKind = '${RecallSessionProjectionKind.LOGICAL_SESSION}')`,
          uniquePartitionField: 'logicalSessionOccurrenceId',
          outputFields: [],
        },
      ];
    default:
      throw new Error('Recall target physical source store responsibility unsupported');
  }
}

function listPhysicalSourceRecordIds(
  collection: ZVecCollection,
  physicalSourceIdentity: string,
  responsibility: RecallGenerationStoreContract['responsibility'],
): string[] {
  if (collection.stats.docCount === 0) {
    return [];
  }
  const recordIds: string[] = [];
  for (const enumeration of createPhysicalSourceRecordEnumerations(
    physicalSourceIdentity,
    responsibility,
  )) {
    visitExactZvecDocuments(collection, enumeration, ({ id }) => recordIds.push(id));
  }
  return recordIds.toSorted((left, right) => left.localeCompare(right));
}

function createPhysicalSourceExpectedMembership(
  recordIds: PhysicalSourceRecordIds,
): RecallPhysicalSourceExpectedMembership {
  return {
    lexicalSource: createRecallPhysicalSourceStoreMembership(recordIds.lexicalSource),
    dense: createRecallPhysicalSourceStoreMembership(recordIds.dense),
    sessionProjection: createRecallPhysicalSourceStoreMembership(recordIds.sessionProjection),
  };
}

function assertPhysicalSourceMembershipMatches(
  actualRecordIds: PhysicalSourceRecordIds,
  expectedMembership: RecallPhysicalSourceExpectedMembership,
  responsibility: string,
): void {
  const actualMembership = createPhysicalSourceExpectedMembership(actualRecordIds);
  for (const store of ['lexicalSource', 'dense', 'sessionProjection'] as const) {
    const actual = actualMembership[store];
    const expected = expectedMembership[store];
    if (
      actual.count !== expected.count ||
      actual.digest !== expected.digest ||
      actual.canaryRecordId !== expected.canaryRecordId
    ) {
      throw new Error(`Recall target incremental ${responsibility} ${store} membership mismatch`);
    }
  }
}

function readPhysicalSourceRecordIds(
  paths: ReturnType<typeof createRecallGenerationComponentPaths>,
  physicalSourceIdentity: string,
): PhysicalSourceRecordIds {
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  const dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  try {
    return {
      lexicalSource: listPhysicalSourceRecordIds(
        lexicalSource,
        physicalSourceIdentity,
        'lexical-source',
      ),
      dense: listPhysicalSourceRecordIds(dense, physicalSourceIdentity, 'dense-evidence'),
      sessionProjection: listPhysicalSourceRecordIds(
        sessionProjection,
        physicalSourceIdentity,
        'session-projection',
      ),
    };
  } finally {
    sessionProjection.closeSync();
    dense.closeSync();
    lexicalSource.closeSync();
  }
}

function verifyPhysicalSourceExpectedMembership(
  lexicalSource: ZVecCollection,
  dense: ZVecCollection,
  sessionProjection: ZVecCollection,
  generationId: string,
  physicalSourceIdentity: string,
): void {
  const physicalProjectionRowId = `projection_${physicalSourceIdentity}`;
  const physicalProjectionRow = sessionProjection.fetchSync({
    ids: [physicalProjectionRowId],
    outputFields: ['projectionJson'],
    includeVector: false,
  })[physicalProjectionRowId];
  if (physicalProjectionRow === undefined) {
    throw new Error(
      'Recall target incremental physical projection missing during membership verification',
    );
  }
  const artifact = parseRecallGenerationPhysicalProjectionArtifact(
    physicalProjectionRow.fields.projectionJson,
    generationId,
  );
  if (artifact.physicalSourceIdentity !== physicalSourceIdentity) {
    throw new Error(
      'Recall target incremental physical projection identity mismatch during membership verification',
    );
  }
  assertPhysicalSourceMembershipMatches(
    {
      lexicalSource: listPhysicalSourceRecordIds(
        lexicalSource,
        physicalSourceIdentity,
        'lexical-source',
      ),
      dense: listPhysicalSourceRecordIds(dense, physicalSourceIdentity, 'dense-evidence'),
      sessionProjection: listPhysicalSourceRecordIds(
        sessionProjection,
        physicalSourceIdentity,
        'session-projection',
      ),
    },
    artifact.expectedMembership,
    'reopened projection',
  );
}

function parseRecoveryRecordIds(value: unknown, fieldName: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`Recall target incremental deletion recovery ${fieldName} invalid`);
  }
  const recordIds: string[] = [];
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new Error(`Recall target incremental deletion recovery ${fieldName} invalid`);
    }
    recordIds.push(item);
  }
  return recordIds.toSorted((left, right) => left.localeCompare(right));
}

async function invokeDeletionTransferFault(
  options: ConfirmedDeletionIncrementalRecallWorkPlanOptions,
  stage: Parameters<NonNullable<IncrementalRecallTransferServices['incrementalTransferFault']>>[0],
): Promise<void> {
  const request = options.confirmedPhysicalSourceDeletion;
  await options.incrementalTransferFault?.(stage, {
    generationId: request.targetGenerationId,
    physicalSourceIdentity: request.physicalSourceIdentity,
    batchIndex: 0,
    evidenceDocumentCount: 0,
  });
}

async function transferConfirmedPhysicalSourceDeletion(
  options: ConfirmedDeletionIncrementalRecallWorkPlanOptions,
): Promise<CommittedIncrementalRecallWorkPlan> {
  const request = options.confirmedPhysicalSourceDeletion;
  const generationDirectory = join(
    options.generation.generationRootDirectory,
    request.targetGenerationId,
  );
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  await coordinateRecallWriteWindow(
    {
      lockPath: options.generation.lockPath,
      allowRecovery: true,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      await assertIncrementalTransferTargetRemainsActive(options, request.targetGenerationId);
      let denseIds: string[];
      let lexicalSourceIds: string[];
      let projectionIds: string[];
      let recovery: Record<string, unknown>;
      if (existsSync(paths.recoveryRecordPath)) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(await readFile(paths.recoveryRecordPath, 'utf8'));
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(
            `Recall target incremental deletion recovery record invalid: ${message}`,
            { cause: error },
          );
        }
        if (
          typeof parsed !== 'object' ||
          parsed === null ||
          !('operation' in parsed) ||
          parsed.operation !== 'delete-physical-source' ||
          !('generationId' in parsed) ||
          parsed.generationId !== request.targetGenerationId ||
          !('physicalSourceIdentity' in parsed) ||
          parsed.physicalSourceIdentity !== request.physicalSourceIdentity
        ) {
          throw new Error('Recall target incremental deletion recovery record mismatch');
        }
        denseIds = parseRecoveryRecordIds(
          'denseIds' in parsed ? parsed.denseIds : undefined,
          'dense IDs',
        );
        lexicalSourceIds = parseRecoveryRecordIds(
          'lexicalSourceIds' in parsed ? parsed.lexicalSourceIds : undefined,
          'lexical/source IDs',
        );
        projectionIds = parseRecoveryRecordIds(
          'projectionIds' in parsed ? parsed.projectionIds : undefined,
          'projection IDs',
        );
        recovery = {
          version: 1,
          generationId: request.targetGenerationId,
          operation: 'delete-physical-source',
          physicalSourceIdentity: request.physicalSourceIdentity,
          denseIds,
          lexicalSourceIds,
          projectionIds,
        };
      } else {
        const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
        const dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
        const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
        try {
          denseIds = listPhysicalSourceRecordIds(
            dense,
            request.physicalSourceIdentity,
            'dense-evidence',
          );
          lexicalSourceIds = listPhysicalSourceRecordIds(
            lexicalSource,
            request.physicalSourceIdentity,
            'lexical-source',
          );
          projectionIds = listPhysicalSourceRecordIds(
            sessionProjection,
            request.physicalSourceIdentity,
            'session-projection',
          );
        } finally {
          sessionProjection.closeSync();
          dense.closeSync();
          lexicalSource.closeSync();
        }
        recovery = {
          version: 1,
          generationId: request.targetGenerationId,
          operation: 'delete-physical-source',
          physicalSourceIdentity: request.physicalSourceIdentity,
          denseIds,
          lexicalSourceIds,
          projectionIds,
        };
      }
      await invokeDeletionTransferFault(options, 'before-recovery-record');
      await writeRecoveryRecord(paths.recoveryRecordPath, recovery);
      await invokeDeletionTransferFault(options, 'after-recovery-record');
      let lexicalSource: ZVecCollection | undefined;
      let dense: ZVecCollection | undefined;
      let sessionProjection: ZVecCollection | undefined;
      let operationError: unknown;
      try {
        lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
        dense = ZVecOpen(paths.denseStorePath);
        sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
        const existingDenseIds = Object.keys(
          dense.fetchSync({ ids: denseIds, outputFields: [], includeVector: false }),
        );
        if (existingDenseIds.length > 0) {
          assertCheckedStatuses(
            'dense deletion',
            existingDenseIds,
            dense.deleteSync(existingDenseIds),
          );
        }
        await invokeDeletionTransferFault(options, 'after-dense-delete');
        const existingLexicalIds = Object.keys(
          lexicalSource.fetchSync({
            ids: lexicalSourceIds,
            outputFields: [],
            includeVector: false,
          }),
        );
        if (existingLexicalIds.length > 0) {
          assertCheckedStatuses(
            'lexical/source deletion',
            existingLexicalIds,
            lexicalSource.deleteSync(existingLexicalIds),
          );
        }
        await invokeDeletionTransferFault(options, 'after-lexical-source-delete');
        const existingProjectionIds = Object.keys(
          sessionProjection.fetchSync({
            ids: projectionIds,
            outputFields: [],
            includeVector: false,
          }),
        );
        if (existingProjectionIds.length > 0) {
          assertCheckedStatuses(
            'projection deletion',
            existingProjectionIds,
            sessionProjection.deleteSync(existingProjectionIds),
          );
        }
        await invokeDeletionTransferFault(options, 'after-projection-delete');
      } catch (error) {
        operationError = error;
      }
      const closeErrors: unknown[] = [];
      for (const store of [sessionProjection, lexicalSource, dense]) {
        try {
          store?.closeSync();
        } catch (error) {
          closeErrors.push(error);
        }
      }
      if (operationError === undefined && closeErrors.length === 0) {
        try {
          await invokeDeletionTransferFault(options, 'after-store-close');
        } catch (error) {
          operationError = error;
        }
      }
      if (operationError !== undefined || closeErrors.length > 0) {
        writeWindow.retainRecoveryRequired();
        throw new AggregateError(
          [operationError, ...closeErrors].filter((error) => error !== undefined),
          'Recall target incremental deletion or close failed',
        );
      }
      const reopenedLexical = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
      const reopenedDense = ZVecOpen(paths.denseStorePath, { readOnly: true });
      const reopenedProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
      try {
        for (const [responsibility, collection, ids] of [
          ['lexical/source', reopenedLexical, lexicalSourceIds],
          ['dense', reopenedDense, denseIds],
          ['projection', reopenedProjection, projectionIds],
        ] as const) {
          const remaining = collection.fetchSync({ ids, outputFields: [], includeVector: false });
          if (Object.keys(remaining).length > 0) {
            throw new Error(
              `Recall target incremental ${responsibility} deletion verification failed`,
            );
          }
        }
        await invokeDeletionTransferFault(options, 'after-reopened-verification');
      } finally {
        reopenedProjection.closeSync();
        reopenedDense.closeSync();
        reopenedLexical.closeSync();
      }
      await rm(paths.recoveryRecordPath);
      await syncRecallDirectory(generationDirectory);
      writeWindow.attestRecoveryCompleted();
      await invokeDeletionTransferFault(options, 'after-recovery-clear');
    },
  );
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 0,
  };
}

async function transferPhysicalSessionProjectionUpdate(
  options: PhysicalSessionProjectionUpdateOptions,
): Promise<IncrementalRecallWorkPlanTransferOutcome> {
  const { projection, workPlan } = options.physicalSessionProjectionUpdate;
  if (projection.generationId !== workPlan.targetGenerationId) {
    throw new Error('Recall target physical projection update generation mismatch');
  }
  const physicalSource = resolveRecallPhysicalSourceIdentity(
    options.generation.sessionsDirectory,
    projection.sourcePath,
  );
  const generationDirectory = join(
    options.generation.generationRootDirectory,
    workPlan.targetGenerationId,
  );
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  const projectionRowId = `projection_${physicalSource.physicalSourceIdentity}`;
  const collection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  let currentFields: Record<string, unknown>;
  try {
    const current = collection.fetchSync({
      ids: [projectionRowId],
      outputFields: [
        'schemaVersion',
        'generationId',
        'projectionKind',
        'physicalSourceIdentity',
        'logicalSessionOccurrenceId',
        'projectionJson',
      ],
      includeVector: false,
    })[projectionRowId];
    if (current === undefined || typeof current.fields.projectionJson !== 'string') {
      throw new Error(
        `Recall target physical projection update row missing for ${physicalSource.physicalSourceIdentity}`,
      );
    }
    let artifact: unknown;
    try {
      artifact = JSON.parse(current.fields.projectionJson);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall target physical projection update artifact invalid for ${physicalSource.physicalSourceIdentity}: ${message}`,
        { cause: error },
      );
    }
    if (!isUnknownRecord(artifact)) {
      throw new Error(
        `Recall target physical projection update artifact invalid for ${physicalSource.physicalSourceIdentity}`,
      );
    }
    const encoded = encodeRecallSessionProjection(projection);
    if (encoded.status !== RecallProjectionEncodingStatus.ENCODED) {
      throw new Error('Recall target incremental physical projection exceeds the bounded payload');
    }
    currentFields = {
      ...current.fields,
      projectionJson: JSON.stringify({
        ...artifact,
        ingestionProjectionPayload: encoded.payload,
      }),
    };
  } finally {
    collection.closeSync();
  }
  const commitWorkPlan =
    options.physicalSessionProjectionUpdate.acknowledgeMarkers === false
      ? { ...workPlan, sourceMarkerIds: [], workItems: [] }
      : workPlan;
  const markerOptions: MarkerIncrementalRecallWorkPlanOptions = {
    generation: options.generation,
    chunkPolicy: options.chunkPolicy,
    loadTokenizer: () => options.loadTokenizer(),
    embeddingProvider: {
      embedDocuments: (documents, signal) =>
        options.embeddingProvider.embedDocuments(documents, signal),
    },
    resolveProjectIdentity: (sessionOrigin) => options.resolveProjectIdentity(sessionOrigin),
    workPlan: commitWorkPlan,
    ...(options.readRange ? { readRange: options.readRange } : {}),
    ...(options.incrementalTransferFault
      ? { incrementalTransferFault: options.incrementalTransferFault }
      : {}),
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.nowEpochMilliseconds ? { nowEpochMilliseconds: options.nowEpochMilliseconds } : {}),
  };
  await commitPreparedTargetTransfer(markerOptions, {
    physicalSource,
    physicalProjection: projection,
    logicalProjections: [],
    lexicalRows: [],
    denseRows: [],
    logicalProjectionRows: [],
    physicalProjectionRow: { id: projectionRowId, fields: currentFields },
  });
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 0,
  };
}

/**
 * Transfers one physical-source request into target lexical/source, dense, and projection stores.
 * Tokenization and document embeddings finish before any exclusive write window begins.
 */
export async function transferIncrementalRecallWorkPlan(
  options: TransferIncrementalRecallWorkPlanOptions,
): Promise<IncrementalRecallWorkPlanTransferOutcome> {
  const generationId = targetGenerationIdForIncrementalTransfer(options);
  await assertConfiguredIncrementalTransferManifest(options, generationId);
  if (options.confirmedPhysicalSourceDeletion !== undefined) {
    return transferConfirmedPhysicalSourceDeletion(options);
  }
  if (options.physicalSessionProjectionUpdate !== undefined) {
    return transferPhysicalSessionProjectionUpdate(options);
  }
  assertSinglePhysicalSourceWorkPlan(options.workPlan);
  const firstMarker = options.workPlan.workItems[0]?.marker;
  if (firstMarker === undefined) {
    throw new Error('Recall target incremental transfer requires at least one marker work item');
  }
  const generationDirectory = join(options.generation.generationRootDirectory, generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  const physicalSource = resolveRecallPhysicalSourceIdentity(
    options.generation.sessionsDirectory,
    firstMarker.physicalSessionPath,
  );
  const recovered = await recoverPendingIncrementalTransfer(
    options,
    physicalSource.physicalSourceIdentity,
  );
  if (recovered !== null) {
    return recovered;
  }
  const current = readCurrentTargetProjections(options, physicalSource.physicalSourceIdentity);
  const currentRecordIds = readPhysicalSourceRecordIds(
    paths,
    physicalSource.physicalSourceIdentity,
  );
  const physicalProjection =
    current.physicalProjection ??
    (await createInitialRecallPhysicalProjection({
      physicalSessionId: physicalSource.physicalSourceIdentity,
      physicalSessionPath: firstMarker.physicalSessionPath,
      generationId,
    }));
  const appendDelta = await readRecallSessionAppendDelta(
    firstMarker.physicalSessionPath,
    physicalProjection,
  );
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    throw new Error(
      `Recall target incremental append requires reconciliation: ${appendDelta.repairReason}`,
    );
  }
  const sourceModifiedAtEpochMilliseconds = Number(
    (await stat(firstMarker.physicalSessionPath, { bigint: true })).mtimeNs / 1_000_000n,
  );
  const nowEpochMilliseconds = options.nowEpochMilliseconds ?? Date.now;
  const initialSchedule = scheduleRecallWorkPlanEligibility({
    workPlan: options.workPlan,
    sourceModifiedAtEpochMilliseconds,
    preparedDocumentCount: 0,
    nowEpochMilliseconds,
  });
  if (!initialSchedule.ready) {
    return createDeferredTransfer(
      initialSchedule.threshold,
      initialSchedule.readyAtEpochMilliseconds,
    );
  }
  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: current.logicalProjections,
    appendDelta,
    markers: options.workPlan.workItems.map(({ marker }) => marker),
    quiescenceObserved:
      initialSchedule.threshold === RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
  });
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    throw new Error(
      `Recall target incremental projection requires reconciliation: ${projected.repairReason}`,
    );
  }
  const markerCheckpoint = mergeRecallMarkerCheckpoint({
    generationId,
    current: projected.physicalProjection.markerCheckpoint,
    coveredMarkerIds: options.workPlan.sourceMarkerIds,
    runtimeSequences: options.workPlan.workItems.map(({ marker }) => ({
      runtimeInstanceId: marker.runtimeInstanceId,
      sequence: marker.runtimeSequence,
    })),
  });
  const projectedPhysical: PhysicalSessionProjection = {
    ...projected.physicalProjection,
    markerCheckpoint,
  };
  const projectedLogical = projected.logicalProjections.map((projection) => ({
    ...projection,
    markerCheckpoint,
  }));
  const eligibleSessions = await Promise.all(
    projectedLogical.map(async (logicalProjection) => {
      const newlyEligibleSpans = spansForLogicalProjection(
        projected.newlyEligibleSpans,
        logicalProjection,
      );
      return {
        graphView: await materializeIncrementalRecallEligibleGraphView({
          physicalProjection: projectedPhysical,
          logicalProjection,
          newlyEligibleSpans,
          appendDelta,
          ...(options.readRange ? { readRange: options.readRange } : {}),
        }),
        logicalProjection,
        newlyEligibleSpans,
      };
    }),
  );
  const tokenizer = await options.loadTokenizer();
  const prepared = await materializePreparedTargetTransfer(
    options,
    physicalSource,
    projectedPhysical,
    projectedLogical,
    currentRecordIds,
    eligibleSessions,
    tokenizer,
  );
  const preparedDocumentCount = prepared.lexicalRows.filter(
    ({ fields }) => fields.recordKind === 'evidence',
  ).length;
  const preparedSchedule = scheduleRecallWorkPlanEligibility({
    workPlan: options.workPlan,
    sourceModifiedAtEpochMilliseconds,
    preparedDocumentCount,
    nowEpochMilliseconds,
  });
  if (!preparedSchedule.ready) {
    return createDeferredTransfer(
      preparedSchedule.threshold,
      preparedSchedule.readyAtEpochMilliseconds,
    );
  }
  const observedCheckpoint = await commitPreparedTargetTransfer(options, prepared);
  await acknowledgeCoveredRecallMarkers(options.workPlan, observedCheckpoint);
  await options.incrementalTransferFault?.('after-marker-acknowledgement', {
    generationId,
    physicalSourceIdentity: physicalSource.physicalSourceIdentity,
    batchIndex: -1,
    evidenceDocumentCount: preparedDocumentCount,
  });
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: preparedDocumentCount,
  };
}
