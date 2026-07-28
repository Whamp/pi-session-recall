import { access, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  coordinateRecallMarkerReplay,
  type RecallMarkerReplayWorkPlan,
} from './coordinate-recall-marker-replay.js';
import { RecallWorkMarkerTrigger } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { decodeRecallActiveGenerationPointer } from './recall-generation-state.js';
import type { RecallWorkMarkerCodecOptions } from './recall-work-marker.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  RECALL_METADATA_SWEEP_CONTINUATION_FILENAME,
  scanRecallSessionMetadata,
  type KnownRecallSessionMetadataSource,
  type RecallSessionMetadataSweepResult,
} from './scan-recall-session-metadata.js';

/** Explicit lightweight paths and lazy import boundary for one short-lived incremental worker. */
export interface RunRecallIncrementalWorkerOptions extends RecallWorkMarkerCodecOptions {
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  controlDirectory: string;
  targetGenerationId: string;
  knownSources?: readonly KnownRecallSessionMetadataSource[];
  loadHeavyDependencies?: () => Promise<void>;
}

/** One worker invocation result, retaining its unacknowledged work plan for later transfer. */
export interface RecallIncrementalWorkerResult {
  workPlan: RecallMarkerReplayWorkPlan;
  metadataSweep: RecallSessionMetadataSweepResult | null;
  heavyDependenciesLoaded: boolean;
}

async function loadRecallIncrementalWorkerDependencies(): Promise<void> {
  await Promise.all([import('@huggingface/tokenizers'), import('@zvec/zvec')]);
}

async function hasRecallMetadataSweepContinuation(controlDirectory: string): Promise<boolean> {
  try {
    await access(joinRecallMetadataContinuationPath(controlDirectory));
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

function joinRecallMetadataContinuationPath(controlDirectory: string): string {
  return resolve(controlDirectory, RECALL_METADATA_SWEEP_CONTINUATION_FILENAME);
}

function workPlanRequestsRecallMetadataSweep(workPlan: RecallMarkerReplayWorkPlan): boolean {
  return workPlan.workItems.some(
    ({ marker }) => marker.trigger.kind === RecallWorkMarkerTrigger.ARRIVAL,
  );
}

function hasRecallIncrementalTransferWork(
  workPlan: RecallMarkerReplayWorkPlan,
  metadataSweep: RecallSessionMetadataSweepResult | null,
): boolean {
  return (
    workPlan.workItems.length > 0 ||
    (metadataSweep !== null &&
      (metadataSweep.observedSessionMetadata.length > 0 ||
        metadataSweep.missingPhysicalSessionIds.length > 0))
  );
}

/** Coordinates one bounded worker pass and loads tokenizer/zvec only after eligible work exists. */
export async function runRecallIncrementalWorker(
  options: RunRecallIncrementalWorkerOptions,
): Promise<RecallIncrementalWorkerResult> {
  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: options.markerSpoolDirectory,
    markerQuarantineDirectory: options.markerQuarantineDirectory,
    targetGenerationId: options.targetGenerationId,
    trustedSessionRoots: options.trustedSessionRoots,
  });
  const shouldSweepMetadata =
    workPlanRequestsRecallMetadataSweep(workPlan) ||
    (await hasRecallMetadataSweepContinuation(options.controlDirectory));
  const metadataSweep = shouldSweepMetadata
    ? await scanRecallSessionMetadata({
        sessionRootDirectory: options.trustedSessionRoots[0] ?? '',
        controlDirectory: options.controlDirectory,
        knownSources: options.knownSources ?? [],
      })
    : null;
  if (!hasRecallIncrementalTransferWork(workPlan, metadataSweep)) {
    return { workPlan, metadataSweep, heavyDependenciesLoaded: false };
  }
  await (options.loadHeavyDependencies ?? loadRecallIncrementalWorkerDependencies)();
  return { workPlan, metadataSweep, heavyDependenciesLoaded: true };
}

async function readRecallWorkerTargetGenerationId(
  activeGenerationPointerPath: string,
): Promise<string | null> {
  try {
    return decodeRecallActiveGenerationPointer(await readFile(activeGenerationPointerPath, 'utf8'))
      .activeGenerationId;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function runRecallIncrementalWorkerExecutable(): Promise<void> {
  const config = await loadRecallConversationConfig();
  const targetGenerationId = await readRecallWorkerTargetGenerationId(
    config.activeGenerationPointerPath,
  );
  if (targetGenerationId === null) {
    return;
  }
  await runRecallIncrementalWorker({
    markerSpoolDirectory: config.markerSpoolDirectory,
    markerQuarantineDirectory: config.markerQuarantineDirectory,
    controlDirectory: config.markerControlDirectory,
    targetGenerationId,
    trustedSessionRoots: [config.sessionsDirectory],
  });
}

const executedModulePath = process.argv[1];
if (
  executedModulePath !== undefined &&
  resolve(executedModulePath) === fileURLToPath(import.meta.url)
) {
  runRecallIncrementalWorkerExecutable().catch((error: unknown) => {
    process.emitWarning(
      `Recall incremental worker failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
    );
    process.exitCode = 1;
  });
}
