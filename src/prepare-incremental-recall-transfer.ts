import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { RecallProjectionEncodingStatus, RecallProjectionRepairReason } from './enums.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { IncrementalRecallEligibleGraphView } from './materialize-incremental-recall-eligible-graph-view.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  encodeRecallSessionProjection,
  mergeRecallMarkerCheckpoint,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallMarkerCheckpoint,
} from './recall-session-projection.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  buildSessionConversationDocuments,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';
import type { IndexedSessionConversationChunk } from './zvec-conversation-store.js';

/** One validated logical graph plus only the spans that crossed recall eligibility now. */
export interface IncrementalRecallEligibleLogicalSession {
  graphView: IncrementalRecallEligibleGraphView;
  logicalProjection: LogicalSessionProjection;
  newlyEligibleSpans: readonly RecallEligibleSourceSpan[];
}

/** Durable physical/logical state that must follow successful evidence writes. */
export interface IncrementalRecallCheckpointIntent {
  readonly physicalProjection: PhysicalSessionProjection;
  readonly logicalProjections: readonly LogicalSessionProjection[];
}

/** Immutable evidence, checkpoint intent, and embedding accounting prepared outside zvec. */
export interface PreparedIncrementalRecallTransfer {
  readonly status: RecallProjectionEncodingStatus.ENCODED;
  readonly targetGenerationId: string;
  readonly documents: readonly IndexedSessionConversationChunk[];
  readonly checkpointIntent: IncrementalRecallCheckpointIntent;
  readonly workPlan: RecallMarkerReplayWorkPlan;
  readonly newlyEmbeddedChunks: number;
  readonly embeddingRequestCount: number;
}

/** Projection overflow result that prevents tokenizer, embedding, lock, or zvec work. */
export interface IncrementalRecallTransferReconciliation {
  readonly status: RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION;
  readonly repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW;
  readonly byteLength: number;
}

/** Prepared transfer or explicit projection-overflow reconciliation. */
export type PrepareIncrementalRecallTransferResult =
  | PreparedIncrementalRecallTransfer
  | IncrementalRecallTransferReconciliation;

/** Source projections and injectable heavy boundaries for one pre-lock transfer preparation. */
export interface PrepareIncrementalRecallTransferOptions {
  physicalProjection: PhysicalSessionProjection;
  eligibleSessions: readonly IncrementalRecallEligibleLogicalSession[];
  workPlan: RecallMarkerReplayWorkPlan;
  chunkPolicy: RecallChunkPolicy;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  resolveProjectIdentity(sessionOrigin: string): Promise<ResolvedProjectIdentity | null>;
  embeddingProvider: Pick<RecallEmbeddingProvider, 'embedDocuments'>;
  maxProjectionPayloadBytes?: number;
  signal?: AbortSignal;
}

function createTransferMarkerCheckpoint(
  physicalProjection: PhysicalSessionProjection,
  workPlan: RecallMarkerReplayWorkPlan,
): RecallMarkerCheckpoint {
  return mergeRecallMarkerCheckpoint({
    generationId: workPlan.targetGenerationId,
    current: physicalProjection.markerCheckpoint,
    coveredMarkerIds: workPlan.sourceMarkerIds,
    runtimeSequences: workPlan.workItems.map(({ marker }) => ({
      runtimeInstanceId: marker.runtimeInstanceId,
      sequence: marker.runtimeSequence,
    })),
  });
}

function createCheckpointIntent(
  options: PrepareIncrementalRecallTransferOptions,
): IncrementalRecallCheckpointIntent {
  const { physicalProjection, eligibleSessions, workPlan } = options;
  if (physicalProjection.generationId !== workPlan.targetGenerationId) {
    throw new Error('Recall incremental preparation generation mismatch');
  }
  for (const { marker } of workPlan.workItems) {
    if (
      marker.physicalSessionId !== physicalProjection.physicalSessionId ||
      marker.physicalSessionPath !== physicalProjection.sourcePath
    ) {
      throw new Error('Recall incremental preparation marker physical session mismatch');
    }
  }
  const projectedLogicalSessionIds = eligibleSessions.map(
    ({ logicalProjection }) => logicalProjection.logicalSessionId,
  );
  if (
    new Set(projectedLogicalSessionIds).size !== projectedLogicalSessionIds.length ||
    physicalProjection.logicalSessionIds.length !== projectedLogicalSessionIds.length ||
    physicalProjection.logicalSessionIds.some(
      (logicalSessionId) => !projectedLogicalSessionIds.includes(logicalSessionId),
    )
  ) {
    throw new Error('Recall incremental preparation logical projection set mismatch');
  }
  for (const { logicalProjection, newlyEligibleSpans } of eligibleSessions) {
    if (logicalProjection.generationId !== workPlan.targetGenerationId) {
      throw new Error('Recall incremental preparation generation mismatch');
    }
    if (
      logicalProjection.physicalProjectionId !== physicalProjection.projectionId ||
      logicalProjection.physicalSessionId !== physicalProjection.physicalSessionId
    ) {
      throw new Error('Recall incremental preparation physical/logical projection mismatch');
    }
    const eligibleContributorEntryIds = new Set(logicalProjection.eligibleContributorEntryIds);
    if (
      newlyEligibleSpans.some(({ contributorEntryIds }) =>
        contributorEntryIds.some(
          (contributorEntryId) => !eligibleContributorEntryIds.has(contributorEntryId),
        ),
      )
    ) {
      throw new Error('Recall incremental preparation span contains an ineligible contributor');
    }
  }
  const markerCheckpoint = createTransferMarkerCheckpoint(physicalProjection, workPlan);
  const updatedPhysicalProjection: PhysicalSessionProjection = {
    ...physicalProjection,
    markerCheckpoint,
  };
  const updatedLogicalProjections = eligibleSessions.map(({ logicalProjection }) =>
    Object.freeze({ ...logicalProjection, markerCheckpoint }),
  );
  return {
    physicalProjection: Object.freeze(updatedPhysicalProjection),
    logicalProjections: Object.freeze(updatedLogicalProjections),
  };
}

function findProjectionOverflow(
  checkpointIntent: IncrementalRecallCheckpointIntent,
  maxProjectionPayloadBytes?: number,
): IncrementalRecallTransferReconciliation | null {
  const encodingOptions =
    maxProjectionPayloadBytes === undefined ? {} : { maxPayloadBytes: maxProjectionPayloadBytes };
  for (const projection of [
    checkpointIntent.physicalProjection,
    ...checkpointIntent.logicalProjections,
  ]) {
    const encoded = encodeRecallSessionProjection(projection, encodingOptions);
    if (encoded.status === RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION) {
      return Object.freeze({
        status: RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION,
        repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW,
        byteLength: encoded.byteLength,
      });
    }
  }
  return null;
}

async function attributeIncrementalRecallDocuments(
  documents: readonly ReturnType<typeof buildSessionConversationDocuments>[number][],
  resolveProjectIdentity: PrepareIncrementalRecallTransferOptions['resolveProjectIdentity'],
): Promise<ReturnType<typeof buildSessionConversationDocuments>> {
  const attributionByOrigin = new Map<string, Promise<ResolvedProjectIdentity | null>>();
  return Promise.all(
    documents.map(async (document) => {
      let resolution = attributionByOrigin.get(document.cwd);
      if (resolution === undefined) {
        resolution = resolveProjectIdentity(document.cwd);
        attributionByOrigin.set(document.cwd, resolution);
      }
      return { ...document, projectAttribution: await resolution };
    }),
  );
}

async function resolvePreparedRecallEmbeddings(
  documents: readonly ReturnType<typeof buildSessionConversationDocuments>[number][],
  options: PrepareIncrementalRecallTransferOptions,
): Promise<{
  documents: IndexedSessionConversationChunk[];
  newlyEmbeddedChunks: number;
  embeddingRequestCount: number;
}> {
  const denseDocuments = documents.filter(({ isDenseSearchable }) => isDenseSearchable);
  const vectors =
    denseDocuments.length === 0
      ? []
      : await options.embeddingProvider.embedDocuments(
          denseDocuments.map(({ content }) => content),
          options.signal,
        );
  if (vectors.length !== denseDocuments.length) {
    throw new Error(
      `Recall incremental preparation embedding response count mismatch: expected ${denseDocuments.length}, received ${vectors.length}`,
    );
  }
  let denseIndex = 0;
  const indexedDocuments: IndexedSessionConversationChunk[] = documents.map((document) => {
    if (!document.isDenseSearchable) {
      return { ...document, isDenseSearchable: false };
    }
    const embedding = vectors[denseIndex];
    denseIndex += 1;
    if (embedding === undefined) {
      throw new Error(`Recall incremental preparation embedding missing for ${document.id}`);
    }
    return { ...document, isDenseSearchable: true, embedding: [...embedding] };
  });
  return {
    documents: indexedDocuments,
    newlyEmbeddedChunks: denseDocuments.length,
    embeddingRequestCount: denseDocuments.length === 0 ? 0 : 1,
  };
}

/**
 * Resolves exact tokenization, project identity, and document embeddings before any write lock.
 * Only documents touched by newly eligible contributors are materialized.
 */
export async function prepareIncrementalRecallTransfer(
  options: PrepareIncrementalRecallTransferOptions,
): Promise<PrepareIncrementalRecallTransferResult> {
  const checkpointIntent = createCheckpointIntent(options);
  const overflow = findProjectionOverflow(checkpointIntent, options.maxProjectionPayloadBytes);
  if (overflow !== null) {
    return overflow;
  }
  const tokenizer = await options.loadTokenizer();
  const builtDocuments = options.eligibleSessions.flatMap(
    ({ graphView, logicalProjection, newlyEligibleSpans }) => {
      if (graphView.logicalSessionId !== logicalProjection.logicalSessionId) {
        throw new Error('Recall incremental preparation logical session identity mismatch');
      }
      const graph = graphView.graph;
      const newlyEligibleContributorEntryIds = new Set(
        newlyEligibleSpans.flatMap(({ contributorEntryIds }) => contributorEntryIds),
      );
      return buildSessionConversationDocuments(
        graph,
        new Set(logicalProjection.eligibleContributorEntryIds),
        {
          sessionPath: graphView.physicalPath,
          logicalSessionIdentity: logicalProjection.logicalSessionId,
          physicalSessionProjectionId: options.physicalProjection.projectionId,
          newlyEligibleContributorEntryIds,
          tokenizer,
          maxTokens: options.chunkPolicy.maxTokens,
          overlapTokens: options.chunkPolicy.overlapTokens,
        },
      );
    },
  );
  const attributedDocuments = await attributeIncrementalRecallDocuments(
    builtDocuments,
    (sessionOrigin) => options.resolveProjectIdentity(sessionOrigin),
  );
  const resolved = await resolvePreparedRecallEmbeddings(attributedDocuments, options);
  const documents = Object.freeze(resolved.documents.map((document) => Object.freeze(document)));
  return Object.freeze({
    status: RecallProjectionEncodingStatus.ENCODED,
    targetGenerationId: options.workPlan.targetGenerationId,
    documents,
    checkpointIntent,
    workPlan: options.workPlan,
    newlyEmbeddedChunks: resolved.newlyEmbeddedChunks,
    embeddingRequestCount: resolved.embeddingRequestCount,
  });
}
