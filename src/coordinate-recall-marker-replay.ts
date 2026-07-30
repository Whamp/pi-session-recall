import { randomUUID } from 'node:crypto';
import { mkdir, readdir, readFile, rename, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { RecallMarkerQuarantineCategory, RecallWorkMarkerTrigger } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  decodeRecallWorkMarker,
  type RecallWorkMarker,
  type RecallWorkMarkerCodecOptions,
} from './recall-work-marker.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Scalar-only count and age for quarantined marker diagnostics. */
export interface RecallMarkerQuarantineDiagnostic {
  category: RecallMarkerQuarantineCategory;
  count: number;
  oldestAgeMilliseconds: number;
}

/** One coalesced replay item plus every exact marker ID it represents. */
export interface RecallMarkerReplayWorkItem {
  marker: RecallWorkMarker;
  coveredMarkerIds: string[];
}

/** Generation-state paths used after ordinary replay proves the retained spool empty. */
export interface RecallGenerationReplayCompletionPaths {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerQuarantineDirectory: string;
  /** Target generation root used to prove fixed replay from physical projections. */
  generationRootDirectory: string;
  lockPath: string;
}

/** Deterministic marker work that remains unacknowledged until a generation checkpoint covers it. */
export interface RecallMarkerReplayWorkPlan {
  targetGenerationId: string;
  markerSpoolDirectory: string;
  retainedMarkerDirectory?: string;
  generationReplayCompletion?: RecallGenerationReplayCompletionPaths;
  discoveredMarkerCount: number;
  sourceMarkerIds: string[];
  workItems: RecallMarkerReplayWorkItem[];
  quarantineDiagnostics: RecallMarkerQuarantineDiagnostic[];
  /** Marker files deliberately excluded because they arrived after the fixed replay snapshot. */
  ordinaryBacklogMarkerCount?: number;
}

/** Spool, generation, trust, and diagnostic-clock boundaries for marker replay coordination. */
export interface CoordinateRecallMarkerReplayOptions extends RecallWorkMarkerCodecOptions {
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  retainedMarkerDirectory?: string;
  generationReplayCompletion?: RecallGenerationReplayCompletionPaths;
  targetGenerationId: string;
  /** Exact marker IDs eligible while the target generation remains replay-pending. */
  fixedReplayMarkerIds?: readonly string[];
  nowEpochMilliseconds?: () => number;
}

interface MutableRecallMarkerReplayWorkItem {
  marker: RecallWorkMarker;
  coveredMarkerIds: string[];
}

const supportedRecallWorkMarkerTriggerKinds = new Set<string>([
  RecallWorkMarkerTrigger.ACTIVITY,
  RecallWorkMarkerTrigger.ARRIVAL,
  RecallWorkMarkerTrigger.BRANCH_EXIT,
  RecallWorkMarkerTrigger.COMPACTION,
  RecallWorkMarkerTrigger.DEPARTURE,
]);

function compareRecallMarkerScalar(left: string, right: string): number {
  if (left < right) {
    return -1;
  }
  if (left > right) {
    return 1;
  }
  return 0;
}

function compareRecallWorkMarkers(left: RecallWorkMarker, right: RecallWorkMarker): number {
  return (
    compareRecallMarkerScalar(left.physicalSessionId, right.physicalSessionId) ||
    compareRecallMarkerScalar(left.runtimeInstanceId, right.runtimeInstanceId) ||
    left.runtimeSequence - right.runtimeSequence ||
    compareRecallMarkerScalar(left.markerId, right.markerId)
  );
}

function addLatestRecallMarker(
  workItemsByKey: Map<string, MutableRecallMarkerReplayWorkItem>,
  key: string,
  marker: RecallWorkMarker,
): void {
  const existing = workItemsByKey.get(key);
  if (existing === undefined) {
    workItemsByKey.set(key, { marker, coveredMarkerIds: [marker.markerId] });
    return;
  }
  existing.coveredMarkerIds.push(marker.markerId);
  if (compareRecallWorkMarkers(existing.marker, marker) < 0) {
    existing.marker = marker;
  }
}

function coalesceRecallWorkMarkers(
  markers: readonly RecallWorkMarker[],
): RecallMarkerReplayWorkItem[] {
  const workItemsByKey = new Map<string, MutableRecallMarkerReplayWorkItem>();
  for (const marker of markers) {
    if (marker.trigger.kind === RecallWorkMarkerTrigger.BRANCH_EXIT) {
      workItemsByKey.set(JSON.stringify(['branch-exit', marker.markerId]), {
        marker,
        coveredMarkerIds: [marker.markerId],
      });
      continue;
    }
    const key =
      marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION
        ? JSON.stringify([
            marker.physicalSessionId,
            'compaction',
            marker.trigger.logicalSessionId,
            marker.trigger.compactionEntryId,
          ])
        : JSON.stringify([marker.physicalSessionId, marker.runtimeInstanceId, marker.trigger.kind]);
    addLatestRecallMarker(workItemsByKey, key, marker);
  }
  return [...workItemsByKey.values()]
    .map(({ marker, coveredMarkerIds }) => ({
      marker,
      coveredMarkerIds: coveredMarkerIds.toSorted(),
    }))
    .toSorted((left, right) => compareRecallWorkMarkers(left.marker, right.marker));
}

interface RecallMarkerDiscoveryResult {
  markers: RecallWorkMarker[];
  quarantineDiagnostics: RecallMarkerQuarantineDiagnostic[];
}

function classifyRecallMarkerQuarantine(source: string): RecallMarkerQuarantineCategory {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    return RecallMarkerQuarantineCategory.CORRUPT;
  }
  if (!isUnknownRecord(value)) {
    return RecallMarkerQuarantineCategory.CORRUPT;
  }
  if (value.version !== undefined && value.version !== 1) {
    return RecallMarkerQuarantineCategory.UNSUPPORTED;
  }
  if (
    isUnknownRecord(value.trigger) &&
    typeof value.trigger.kind === 'string' &&
    !supportedRecallWorkMarkerTriggerKinds.has(value.trigger.kind)
  ) {
    return RecallMarkerQuarantineCategory.UNSUPPORTED;
  }
  return RecallMarkerQuarantineCategory.CORRUPT;
}

async function quarantineRecallMarker(
  markerPath: string,
  markerName: string,
  category: RecallMarkerQuarantineCategory,
  markerQuarantineDirectory: string,
): Promise<void> {
  const categoryDirectory = join(markerQuarantineDirectory, category);
  await mkdir(categoryDirectory, { recursive: true });
  await rename(markerPath, join(categoryDirectory, `${markerName}.${randomUUID()}`));
  await syncRecallDirectory(categoryDirectory);
  await syncRecallDirectory(dirname(markerPath));
}

async function readRecallMarkerSpoolNames(markerSpoolDirectory: string): Promise<string[]> {
  try {
    const entries = await readdir(markerSpoolDirectory, { withFileTypes: true });
    return entries
      .filter(
        (entry) => entry.isFile() && !entry.name.startsWith('.') && entry.name.endsWith('.json'),
      )
      .map((entry) => entry.name)
      .toSorted();
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

async function discoverRecallWorkMarkers(
  options: CoordinateRecallMarkerReplayOptions,
  markerNames: readonly string[],
): Promise<RecallMarkerDiscoveryResult> {
  const markersById = new Map<string, RecallWorkMarker>();
  const quarantineByCategory = new Map<
    RecallMarkerQuarantineCategory,
    RecallMarkerQuarantineDiagnostic
  >();
  const nowEpochMilliseconds = options.nowEpochMilliseconds?.() ?? Date.now();
  for (const markerName of markerNames) {
    const markerPath = join(options.markerSpoolDirectory, markerName);
    const source = await readFile(markerPath, 'utf8');
    let marker: RecallWorkMarker | undefined;
    try {
      const decoded = await decodeRecallWorkMarker(source, options);
      if (markerName === `${decoded.markerId}.json`) {
        marker = decoded;
      }
    } catch {
      // The strict decoder intentionally does not expose untrusted marker details.
    }
    if (marker !== undefined) {
      markersById.set(marker.markerId, marker);
      continue;
    }
    const category = classifyRecallMarkerQuarantine(source);
    const markerMetadata = await stat(markerPath);
    await quarantineRecallMarker(
      markerPath,
      markerName,
      category,
      options.markerQuarantineDirectory,
    );
    const ageMilliseconds = Math.max(0, nowEpochMilliseconds - markerMetadata.mtimeMs);
    const diagnostic = quarantineByCategory.get(category);
    if (diagnostic === undefined) {
      quarantineByCategory.set(category, {
        category,
        count: 1,
        oldestAgeMilliseconds: ageMilliseconds,
      });
    } else {
      diagnostic.count += 1;
      diagnostic.oldestAgeMilliseconds = Math.max(
        diagnostic.oldestAgeMilliseconds,
        ageMilliseconds,
      );
    }
  }
  return {
    markers: [...markersById.values()].toSorted(compareRecallWorkMarkers),
    quarantineDiagnostics: [...quarantineByCategory.values()].toSorted((left, right) =>
      compareRecallMarkerScalar(left.category, right.category),
    ),
  };
}

/** Discovers strict markers and returns deterministic coalesced work without loading session bodies. */
export async function coordinateRecallMarkerReplay(
  options: CoordinateRecallMarkerReplayOptions,
): Promise<RecallMarkerReplayWorkPlan> {
  const allMarkerNames = await readRecallMarkerSpoolNames(options.markerSpoolDirectory);
  const fixedReplayMarkerIds =
    options.fixedReplayMarkerIds === undefined ? null : new Set(options.fixedReplayMarkerIds);
  const markerNames =
    fixedReplayMarkerIds === null
      ? allMarkerNames
      : allMarkerNames.filter((markerName) =>
          fixedReplayMarkerIds.has(markerName.slice(0, -'.json'.length)),
        );
  const discovery = await discoverRecallWorkMarkers(options, markerNames);
  return {
    targetGenerationId: options.targetGenerationId,
    markerSpoolDirectory: options.markerSpoolDirectory,
    ...(options.retainedMarkerDirectory
      ? { retainedMarkerDirectory: options.retainedMarkerDirectory }
      : {}),
    ...(options.generationReplayCompletion
      ? { generationReplayCompletion: options.generationReplayCompletion }
      : {}),
    discoveredMarkerCount: discovery.markers.length,
    sourceMarkerIds: discovery.markers.map(({ markerId }) => markerId),
    workItems: coalesceRecallWorkMarkers(discovery.markers),
    quarantineDiagnostics: discovery.quarantineDiagnostics,
    ordinaryBacklogMarkerCount: allMarkerNames.length - markerNames.length,
  };
}
